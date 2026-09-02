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


# ---- small-model (Phase 3 forward) shared spec --------------------------
MODEL_D, MODEL_HIDDEN, MODEL_HEADS, MODEL_LAYERS = 32, 64, 4, 2
MODEL_PATCH_LEN, MODEL_OUT_LEN, MODEL_QUANTILES = 4, 8, (0.25, 0.5, 0.75)


def _model_weights(rng) -> dict[str, np.ndarray]:
    hd = MODEL_D // MODEL_HEADS
    rb_in = 2 * (MODEL_PATCH_LEN + MODEL_OUT_LEN)  # 24
    w = {
        "pre_transformer_resblock.hidden_layer.weight": _linear(rng, MODEL_D, rb_in),
        "pre_transformer_resblock.output_layer.weight": _linear(rng, MODEL_D, MODEL_D),
        "pre_transformer_resblock.residual_layer.weight": _linear(rng, MODEL_D, rb_in),
        "output_head.weight": _linear(rng, MODEL_OUT_LEN * len(MODEL_QUANTILES), MODEL_D),
        "output_head.bias": _normal(rng, MODEL_OUT_LEN * len(MODEL_QUANTILES), scale=0.01),
    }
    for i in range(MODEL_LAYERS):
        w |= {
            f"transformer_stack.layers.{i}.{k}": v
            for k, v in _layer_weights(rng, MODEL_D, hd, MODEL_HIDDEN).items()
        }
    return w


def _model_inputs(cpm: bool):
    """(2, 2, 6, 4) patched inputs: interior noise masks, a leading fully
    masked patch (exercises cumprod leading-only semantics), one covariate
    variate; optionally a horizon-style CPM mask."""

    def gen(rng):
        values = _normal(rng, 2, 2, 6, MODEL_PATCH_LEN)
        masks = rng.random((2, 2, 6, MODEL_PATCH_LEN)) < 0.1
        masks[:, :, 0, :] = True  # leading fully-masked patch
        patch_is_target = np.ones((2, 2, 6), dtype=bool)
        patch_is_target[:, 1, :] = False  # variate 1 = covariate
        out = {
            "values": values,
            "masks": masks,
            "patch_is_target": patch_is_target,
        }
        if cpm:
            patch_cpm = np.zeros((2, 6), dtype=bool)
            patch_cpm[0, 4:] = True
            patch_cpm[1, 5] = True
            out["patch_cpm_mask"] = patch_cpm
        return out

    return gen


def _real_forward_inputs(rng):
    """Fixture-flavoured patched inputs at REAL dims for the real-weights
    forward case: b=2, v=2 (variate 0 target, variate 1 covariate),
    ctx 512 + horizon 128 = 640 pts = 20 patches of 32.

    Structure mirrors the small-model case: one leading fully-masked patch
    (cumprod leading-only semantics), sparse interior masks in the context,
    CPM mask over the 4 horizon patches.
    """
    n = 640
    t = np.arange(n, dtype=np.float64)
    b0_target = 2.0 + 0.01 * t + rng.normal(0.0, 0.1, n)
    b0_cov = np.sin(2 * np.pi * t / 24.0)
    b1_target = np.sin(2 * np.pi * t / 32.0) + rng.normal(0.0, 0.05, n)
    b1_cov = rng.normal(0.0, 1.0, n)
    values = np.stack([[b0_target, b0_cov], [b1_target, b1_cov]]).astype(np.float32)
    values = values.reshape(2, 2, 20, 32)

    masks = np.zeros((2, 2, 20, 32), dtype=bool)
    masks[:, :, 0, :] = True  # leading fully-masked patch (left padding)
    masks[:, :, 1:16, :] |= rng.random((2, 2, 15, 32)) < 0.02

    patch_is_target = np.zeros((2, 2, 20), dtype=bool)
    patch_is_target[:, 0, :] = True

    patch_cpm_mask = np.zeros((2, 20), dtype=bool)
    patch_cpm_mask[:, 16:] = True
    return {
        "values": values,
        "masks": masks,
        "patch_is_target": patch_is_target,
        "patch_cpm_mask": patch_cpm_mask,
    }


@dataclass(frozen=True)
class Case:
    """One parity case: generators + gate (from SPEC A1)."""

    inputs: Callable[[np.random.Generator], dict[str, np.ndarray]]
    weights: Callable[[np.random.Generator], dict[str, np.ndarray]] | None = None
    tol: float | None = None
    bit_exact: bool = False
    # Per-output overrides of `tol`: value = max abs diff; None = bit-exact.
    per_output_tol: dict[str, float | None] = field(default_factory=dict)
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
    # ---- Phase 3: full forward (small random-weights model) ----
    "model_forward_small": Case(
        inputs=_model_inputs(cpm=True),
        weights=_model_weights,
        tol=ACTIVATION_TOL,
        per_output_tol={"revin_mu": NORM_TOL, "revin_sigma": NORM_TOL,
                        # additive masks: integer logic, must match exactly
                        "seq_mask_0": None, "seq_mask_1": None},
        meta={"freeze_after": None},
    ),
    "model_forward_freeze": Case(
        # no CPM path; exercises the freeze_after stat-freeze branch
        inputs=_model_inputs(cpm=False),
        weights=_model_weights,
        tol=ACTIVATION_TOL,
        per_output_tol={"revin_mu": NORM_TOL, "revin_sigma": NORM_TOL,
                        "seq_mask_0": None, "seq_mask_1": None},
        meta={"freeze_after": 2},
    ),
    # ---- real checkpoint weights (20 layers, d=1280, 330.7M params) ----
    # No generated weights: both runners read
    # models/timesfm_3_0/original/model.safetensors themselves (avoids a
    # 1.3 GB npz copy; the file itself is the shared bit-identical seed).
    # Torch side keeps use_sdpa=False + forced PARITY_EPS (PLAN Phase 2
    # review note: this run does NOT claim official-default-kernel parity).
    # aux_transformer_output gate is scale-adjusted, MEASURED (torch 2.13 /
    # mlx 0.32.2, both CPU): residual stream reaches |x| ≈ 223 (ulp = 1.5e-5);
    # stack INPUT already differs by 1 ulp (kernel reduction orders), the
    # output by 2.4e-4 ≈ 16 ulp after 20 layers — pure fp32 accumulation, a
    # code error would scale with structure, not with |residual|. Final logits
    # (data scale, post-inverse-RevIN) hold the plain 1e-4 gate at 2.7e-5.
    "model_forward_real": Case(
        inputs=_real_forward_inputs,
        weights=None,
        tol=ACTIVATION_TOL,
        per_output_tol={"revin_mu": NORM_TOL, "revin_sigma": NORM_TOL,
                        "aux_transformer_output": 1e-3,  # ≈ 4.5e-6 relative
                        **{f"seq_mask_{i}": None for i in range(20)}},
        meta={"weights": "checkpoint", "layers": 20},
    ),
}
