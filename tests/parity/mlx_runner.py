"""Parity runner for the MLX stack. Runs in the main .venv (pytest process or
`uv run python`). Forces CPU (SPEC A1), applies the shared random state dict to
OUR port with strict loading (R6), and stamps environment metadata into the
output npz, mirroring `torch_runner.py`.
"""

from __future__ import annotations

import json

import mlx.core as mx
import mlx.nn as mlx_nn  # noqa: F401  (imported to fail loudly if mlx missing)
import numpy as np
from mlx.utils import tree_flatten

import cases as case_specs
from cases import PARITY_EPS, REAL_D, REAL_HEADS, SMALL_D, SMALL_HEADS, SMALL_HIDDEN
from mlx_timesfm.config import TransformerConfig
from mlx_timesfm.normalization import PerDimScale, RMSNorm
from mlx_timesfm.transformer import (
    MixingTransformer,
    MultiHeadAttention,
    StackedMixingTransformer,
    make_attn_mask,
    make_segment_mask,
    rope,
)
from mlx_timesfm.util import load_parameters


def _a(arrs: dict[str, np.ndarray], k: str) -> mx.array:
    return mx.array(arrs[k])


def _eval_out(out: dict[str, mx.array | np.ndarray]) -> dict[str, np.ndarray]:
    arrays = {k: v for k, v in out.items() if isinstance(v, mx.array)}
    mx.eval(*arrays.values())
    return {k: (np.asarray(v) if isinstance(v, mx.array) else v) for k, v in out.items()}


def _mlx_transformer_config(d: int, hidden: int, heads: int) -> TransformerConfig:
    return TransformerConfig(
        model_dims=d,
        hidden_dims=hidden,
        num_heads=heads,
        attention_norm="rms",
        feedforward_norm="rms",
        qk_norm="rms",
        use_bias=False,
        use_rope_seq=True,
        use_rope_var=False,
        ff_activation="relu",
        deterministic=True,
    )


def case_relu(i, w):
    return {"y": mx.maximum(_a(i, "x"), 0)}


def case_rmsnorm(i, w):
    m = RMSNorm(w["weight"].shape[0], eps=PARITY_EPS)
    load_parameters(m, {k: mx.array(v) for k, v in w.items()})
    return {"y": m(_a(i, "x"))}


def case_per_dim_scale(i, w):
    m = PerDimScale(w["per_dim_scale"].shape[0])
    load_parameters(m, {k: mx.array(v) for k, v in w.items()})
    return {"y": m(_a(i, "x"))}


def case_rope_3d(i, w):
    return {"y": rope(_a(i, "x"), _a(i, "position"))}


def case_rope_4d(i, w):
    return {"y": rope(_a(i, "x"), _a(i, "position"))}


def _mask_case(i, w):
    return {
        "mask": make_attn_mask(
            query_length=int(i["q_len"]),
            num_all_masked_kv=_a(i, "num_all_masked_kv"),
            query_index_offset=_a(i, "query_index_offset"),
            kv_length=int(i["kv_len"]),
            causal=bool(i["causal"]),
        )
    }


def case_segment_mask(i, w):
    return {"mask": make_segment_mask(_a(i, "segment_ids"))}


def _build_mha(case: str, w) -> MultiHeadAttention:
    m = MultiHeadAttention(
        num_heads=REAL_HEADS,
        in_features=REAL_D,
        eps=PARITY_EPS,
        use_per_dim_scale=True,
        use_rotary_position_embeddings=(case == "mha_seq"),
        causal_attention=(case == "mha_seq"),
        use_bias=False,
        qk_norm="rms",
        v_norm="none",
    )
    load_parameters(m, {k: mx.array(v) for k, v in w.items()})
    return m


def case_mha_seq(i, w):
    m = _build_mha("mha_seq", w)
    out, mask = m(
        _a(i, "x"),
        segment_ids=_a(i, "segment_ids"),
        segment_pos=_a(i, "segment_pos"),
        patch_mask=_a(i, "patch_mask"),
    )
    return {"y": out, "mask": mask}


def case_mha_var(i, w):
    m = _build_mha("mha_var", w)
    out, mask = m(_a(i, "x"), patch_mask=_a(i, "patch_mask"))
    return {"y": out, "mask": mask}


def case_mixing_layer(i, w):
    m = MixingTransformer(
        _mlx_transformer_config(REAL_D, REAL_D, REAL_HEADS),
        eps=PARITY_EPS,
        use_variate_attention=True,
    )
    load_parameters(m, {k: mx.array(v) for k, v in w.items()})
    out, mask = m(
        _a(i, "x"),
        _a(i, "patch_mask"),
        segment_ids=_a(i, "segment_ids"),
        segment_pos=_a(i, "segment_pos"),
    )
    return {"y": out, "mask": mask}


def case_stacked_small(i, w):
    m = StackedMixingTransformer(
        case_specs.SMALL_LAYERS,
        _mlx_transformer_config(SMALL_D, SMALL_HIDDEN, SMALL_HEADS),
        eps=PARITY_EPS,
        use_variate_attention=True,
    )
    load_parameters(m, {k: mx.array(v) for k, v in w.items()})
    out, _ = m(_a(i, "x"), _a(i, "patch_mask"))
    return {"y": out}


def case_stack_keys(i, w):
    m = StackedMixingTransformer(
        20, _mlx_transformer_config(REAL_D, REAL_D, REAL_HEADS), eps=PARITY_EPS
    )
    entries = sorted(
        f"{k}|{','.join(map(str, v.shape))}" for k, v in tree_flatten(m.parameters())
    )
    return {"keys": np.array("\n".join(entries))}


CASE_RUNNERS = {
    "relu": case_relu,
    "rmsnorm": case_rmsnorm,
    "per_dim_scale": case_per_dim_scale,
    "rope_3d": case_rope_3d,
    "rope_4d": case_rope_4d,
    "attn_mask": _mask_case,
    "attn_mask_nc": _mask_case,
    "segment_mask": case_segment_mask,
    "mha_seq": case_mha_seq,
    "mha_var": case_mha_var,
    "mixing_layer": case_mixing_layer,
    "stacked_small": case_stacked_small,
    "stack_keys": case_stack_keys,
}


def run_mlx(case: str, inputs_path: str, weights_path: str, outputs_path: str) -> dict:
    mx.set_default_device(mx.cpu)

    data = dict(np.load(inputs_path))
    weights = dict(np.load(weights_path)) if weights_path != "-" else {}
    out = CASE_RUNNERS[case](data, weights)
    out = _eval_out(out)

    meta = {
        "stack": "mlx",
        "mlx_version": mx.__version__,
        "effective_rmsnorm_eps": PARITY_EPS,
        "attention_path": "port manual (Q pre-scaled by sqrt(head_dim))",
        "device": "cpu",
    }
    np.savez(
        outputs_path,
        **{k: np.ascontiguousarray(v) for k, v in out.items()},
        _meta=np.array(json.dumps(meta)),
    )
    return {"op": case, "outputs": sorted(out), "meta": meta}
