"""Shared parity case specs — PURE NUMPY (no framework imports).

Importable from BOTH environments (main .venv and .venv-torch). Each case
defines deterministic input + random-weight generation and its A1 gate; the
framework-specific application lives in ``torch_runner.py`` / ``mlx_runner.py``
under the same case names. Both sides load the SAME random state dict with
strict checking, which itself pins our attribute naming to the reference (R6).

Weight names here are reference state-dict names, e.g.
``seq_attn.per_dim_scale.per_dim_scale``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Explicit RMSNorm eps used on BOTH stacks in parity (SPEC R5). Tests pin this
# to mlx_timesfm.config.INFERENCE_RMSNORM_EPS so inference and parity agree
# until the 1b probe settles the reference's effective eps.
PARITY_EPS = 1e-5

# Real checkpoint dims for op-level cases (d=1280, heads=16 → head_dim=80).
REAL_D, REAL_HEADS = 1280, 16
REAL_HD = REAL_D // REAL_HEADS
# Small dims for the ×20 structural case (fast; shape structure, not scale).
SMALL_D, SMALL_HEADS, SMALL_HIDDEN, SMALL_LAYERS = 64, 4, 64, 20


def _normal(rng, *shape, scale=1.0):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def _tail_mask(rng, rows: int, n: int) -> np.ndarray:
    """Random trailing-masked rows (True=masked); every causal query row keeps
    at least one attendable KV (m ≤ n//3 leaves j=0.. alive)."""
    m = np.zeros((rows, n), dtype=bool)
    for r in range(rows):
        k = int(rng.integers(0, max(1, n // 3) + 1))
        m[r, n - k :] = True
    return m


def _linear(rng, out_f, in_f):
    return _normal(rng, out_f, in_f, scale=0.02)  # ~ torch 1/sqrt(fan_in) init


def _ln_weight(rng, d):
    return (_normal(rng, d, scale=0.05) + 1.0).astype(np.float32)


def _mha_weights(rng, d: int, hd: int, prefix: str = "") -> dict[str, np.ndarray]:
    w = {f"{prefix}{p}_proj.weight": _linear(rng, d, d) for p in ("query", "key", "value", "out")}
    w[f"{prefix}query_ln.weight"] = _ln_weight(rng, hd)
    w[f"{prefix}key_ln.weight"] = _ln_weight(rng, hd)
    w[f"{prefix}per_dim_scale.per_dim_scale"] = _normal(rng, hd, scale=0.5)
    return w


def _layer_weights(rng, d: int, hd: int, hidden: int) -> dict[str, np.ndarray]:
    w = {
        "pre_seq_attn_ln.weight": _ln_weight(rng, d),
        "post_seq_attn_ln.weight": _ln_weight(rng, d),
        "pre_var_attn_ln.weight": _ln_weight(rng, d),
        "post_var_attn_ln.weight": _ln_weight(rng, d),
        "pre_ff_ln.weight": _ln_weight(rng, d),
        "post_ff_ln.weight": _ln_weight(rng, d),
        "ff0.weight": _linear(rng, hidden, d),
        "ff1.weight": _linear(rng, d, hidden),
    }
    w |= {f"seq_attn.{k}": v for k, v in _mha_weights(rng, d, hd).items()}
    w |= {f"var_attn.{k}": v for k, v in _mha_weights(rng, d, hd).items()}
    return w


@dataclass(frozen=True)
class Case:
    """One parity case: generators + gate (from SPEC A1)."""

    inputs: Callable[[np.random.Generator], dict[str, np.ndarray]]
    weights: Callable[[np.random.Generator], dict[str, np.ndarray]] | None = None
    tol: float | None = None
    bit_exact: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


ACTIVATION_TOL = 1e-4  # A1: activations
NORM_TOL = 1e-5  # A1: normalized/reduced quantities

CASES: dict[str, Case] = {
    # ---- smoke (1a) ----
    "relu": Case(
        inputs=lambda rng: {
            "x": np.concatenate([rng.normal(size=4095), np.zeros(1)]).astype(np.float32)
        },
        bit_exact=True,
    ),
    # ---- pointwise (1b) ----
    "rmsnorm": Case(
        inputs=lambda rng: {"x": _normal(rng, 4, 300, REAL_D)},
        weights=lambda rng: {"weight": _ln_weight(rng, REAL_D)},
        tol=NORM_TOL,
    ),
    "per_dim_scale": Case(
        inputs=lambda rng: {"x": _normal(rng, 2, 8, REAL_HEADS, REAL_HD)},
        weights=lambda rng: {"per_dim_scale": _normal(rng, REAL_HD, scale=0.5)},
        tol=NORM_TOL,
    ),
    # Position range 0..299 (patch indices at ctx≤~19k). Cross-framework powf
    # for the RoPE timescale differs by a few ulp; the phase error amplifies
    # ~linearly with |pos| (measured ≈ pos × 1e-7), so pos~1000 sits at the
    # 1e-4 A1 edge even with identical code. End-to-end ranges are covered by
    # A2 (GPU, 2e-3·σ). This gate tests the MECHANISM (arbitrary per-batch
    # positions, SPEC R3), not libm powf agreement.
    "rope_3d": Case(
        inputs=lambda rng: {
            "x": _normal(rng, 3, 12, REAL_HD),
            "position": rng.integers(0, 300, (3, 12)),
        },
        tol=ACTIVATION_TOL,
    ),
    "rope_4d": Case(
        inputs=lambda rng: {
            "x": _normal(rng, 2, 12, REAL_HEADS, REAL_HD),
            # arbitrary per-batch positions (SPEC R3)
            "position": rng.integers(0, 300, (2, 12)),
        },
        tol=ACTIVATION_TOL,
    ),
    # ---- masks (1b R2) — exact integer logic, bit-exact expected ----
    "attn_mask": Case(
        inputs=lambda rng: {
            "num_all_masked_kv": rng.integers(0, 4, (4,)),
            "query_index_offset": rng.integers(0, 6, (4,)),
            "q_len": np.int64(7),
            "kv_len": np.int64(9),
            "causal": np.bool_(True),
        },
        bit_exact=True,
    ),
    "attn_mask_nc": Case(
        inputs=lambda rng: {
            "num_all_masked_kv": rng.integers(0, 4, (4,)),
            "query_index_offset": rng.integers(0, 6, (4,)),
            "q_len": np.int64(7),
            "kv_len": np.int64(9),
            "causal": np.bool_(False),
        },
        bit_exact=True,
    ),
    "segment_mask": Case(
        inputs=lambda rng: {"segment_ids": rng.integers(0, 3, (3, 10))},
        bit_exact=True,
    ),
    # ---- MHA (1c R1/R2/R3 combined; real checkpoint dims) ----
    "mha_seq": Case(
        inputs=lambda rng: {
            "x": _normal(rng, 2, 16, REAL_D),
            "patch_mask": _tail_mask(rng, 2, 16),
            "segment_ids": rng.integers(0, 3, (2, 16)),
            "segment_pos": rng.integers(0, 100, (2, 16)),
        },
        weights=lambda rng: _mha_weights(rng, REAL_D, REAL_HD),
        tol=ACTIVATION_TOL,
    ),
    "mha_var": Case(
        inputs=lambda rng: {
            "x": _normal(rng, 2, 16, REAL_D),
            "patch_mask": _tail_mask(rng, 2, 16),
        },
        weights=lambda rng: _mha_weights(rng, REAL_D, REAL_HD),
        tol=ACTIVATION_TOL,
    ),
    # ---- Phase 2: one full layer (real dims) + ×20 stack (small dims) ----
    "mixing_layer": Case(
        inputs=lambda rng: {
            "x": _normal(rng, 1, 3, 8, REAL_D),
            "patch_mask": _tail_mask(rng, 3, 8).reshape(1, 3, 8),
            "segment_ids": rng.integers(0, 3, (1, 8)),
            "segment_pos": rng.integers(0, 50, (1, 8)),
        },
        weights=lambda rng: _layer_weights(rng, REAL_D, REAL_HD, REAL_D),
        tol=ACTIVATION_TOL,
    ),
    "stacked_small": Case(
        inputs=lambda rng: {
            "x": _normal(rng, 1, 2, 6, SMALL_D),
            "patch_mask": _tail_mask(rng, 2, 6).reshape(1, 2, 6),
        },
        weights=lambda rng: {
            f"layers.{i}.{k}": v
            for i in range(SMALL_LAYERS)
            for k, v in _layer_weights(rng, SMALL_D, SMALL_D // SMALL_HEADS, SMALL_HIDDEN).items()
        },
        tol=ACTIVATION_TOL,
        meta={"layers": SMALL_LAYERS},
    ),
    # structural only: torch dumps its 20-layer state-dict key tree as a string
    "stack_keys": Case(inputs=lambda rng: {}, bit_exact=True, meta={"layers": SMALL_LAYERS}),
}
