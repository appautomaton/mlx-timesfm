#!/usr/bin/env python
"""Parity runner for the PyTorch REFERENCE stack (real timesfm3 modules).

Runs ONLY under `.venv-torch/bin/python` (torch lives there; the main .venv and
the package must never see torch). Invoked as a subprocess by `bridge.py`:

    .venv-torch/bin/python tests/parity/torch_runner.py \
        <case> <inputs.npz> <weights.npz|-> <outputs.npz>

Forces CPU (SPEC A1: a diff must implicate code, not kernels). Loads the shared
random state dict into the REFERENCE modules with strict=True — our port's
attribute naming is pinned by the same dict loading cleanly on both sides.

Attention cases run the reference's MANUAL branch (use_sdpa=False,
rescale_logits=False → net scale √head_dim), mirroring the port (see
transformer.py docstring: mx.fast sdpa deviates ~3e-3 from manual on CPU).

RMSNorm eps is force-set on every torch.nn.RMSNorm to PARITY_EPS so neither
side's default can leak into the diff (SPEC R5; torch 2.13 default is None).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from timesfm3 import configs, model, normalization, transformer

import cases as case_specs
from cases import (
    CASES,
    PARITY_EPS,
    REAL_D,
    REAL_HEADS,
    SMALL_D,
    SMALL_HEADS,
    SMALL_HIDDEN,
    SMALL_LAYERS,
)

torch.set_grad_enabled(False)


def _load_npz(path: str) -> dict[str, np.ndarray]:
    return dict(np.load(path)) if path != "-" else {}


def _t(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a)


def _sd(w: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v) for k, v in w.items()}


def _force_eps(module: torch.nn.Module, eps: float) -> None:
    for m in module.modules():
        if isinstance(m, torch.nn.RMSNorm):
            m.eps = eps


def _to_additive(mask: torch.Tensor) -> np.ndarray:
    return torch.where(mask, torch.tensor(0.0), torch.tensor(-1e9)).numpy()


def _torch_transformer_config(d: int, hidden: int, heads: int) -> configs.TransformerConfig:
    return configs.TransformerConfig(
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
        v_norm="none",
        causal_attention=True,
        use_memory_efficient_attention=True,
        use_sdpa=False,  # manual branch — mirrors the port's default path
    )


# ---------------------------------------------------------------------------
# case builders: (inputs, weights) -> {output_name: ndarray}
# ---------------------------------------------------------------------------


def case_relu(i, w):
    return {"y": torch.nn.functional.relu(_t(i["x"])).numpy()}


def case_rmsnorm(i, w):
    m = torch.nn.RMSNorm(w["weight"].shape[0], eps=PARITY_EPS)
    m.load_state_dict(_sd(w))
    return {"y": m(_t(i["x"])).numpy()}


def case_per_dim_scale(i, w):
    m = normalization.PerDimScale(w["per_dim_scale"].shape[0])
    m.load_state_dict(_sd(w))
    return {"y": m(_t(i["x"])).numpy()}


def case_rope_3d(i, w):
    m = transformer.RotaryPositionalEmbedding(REAL_D // REAL_HEADS)
    return {"y": m(_t(i["x"]), _t(i["position"])).numpy()}


def case_rope_4d(i, w):
    m = transformer.RotaryPositionalEmbedding(REAL_D // REAL_HEADS)
    return {"y": m(_t(i["x"]), _t(i["position"])).numpy()}


def _mask_case(i, w):
    mask = transformer.make_attn_mask(
        query_length=int(i["q_len"]),
        num_all_masked_kv=_t(i["num_all_masked_kv"]),
        query_index_offset=_t(i["query_index_offset"]),
        kv_length=int(i["kv_len"]),
        causal=bool(i["causal"]),
    )
    return {"mask": _to_additive(mask)}


def case_segment_mask(i, w):
    mask = transformer.make_segment_mask(_t(i["segment_ids"]))
    return {"mask": _to_additive(mask)}


def _build_mha(case: str, w) -> transformer.MultiHeadAttention:
    m = transformer.MultiHeadAttention(
        num_heads=REAL_HEADS,
        in_features=REAL_D,
        use_per_dim_scale=True,
        use_rotary_position_embeddings=(case == "mha_seq"),
        causal_attention=(case == "mha_seq"),
        use_bias=False,
        qk_norm="rms",
        v_norm="none",
        use_sdpa=False,
        rescale_logits=False,  # MEA-equivalent: net scale √head_dim
    )
    m.load_state_dict(_sd(w), strict=True)
    m.eval()
    _force_eps(m, PARITY_EPS)
    return m


def case_mha_seq(i, w):
    m = _build_mha("mha_seq", w)
    out, _, mask = m(
        _t(i["x"]),
        segment_ids=_t(i["segment_ids"]),
        segment_pos=_t(i["segment_pos"]),
        patch_mask=_t(i["patch_mask"]),
    )
    return {"y": out.numpy(), "mask": _to_additive(mask)}


def case_mha_var(i, w):
    m = _build_mha("mha_var", w)
    out, _, mask = m(_t(i["x"]), patch_mask=_t(i["patch_mask"]))
    return {"y": out.numpy(), "mask": _to_additive(mask)}


def case_mixing_layer(i, w):
    m = transformer.MixingTransformer(
        _torch_transformer_config(REAL_D, REAL_D, REAL_HEADS),
        use_variate_attention=True,
    )
    m.load_state_dict(_sd(w), strict=True)
    m.eval()
    _force_eps(m, PARITY_EPS)
    out, _, mask = m(
        _t(i["x"]),
        _t(i["patch_mask"]),
        segment_ids=_t(i["segment_ids"]),
        segment_pos=_t(i["segment_pos"]),
    )
    return {"y": out.numpy(), "mask": _to_additive(mask)}


def case_stacked_small(i, w):
    m = transformer.StackedMixingTransformer(
        configs.StackedTransformersConfig(
            num_layers=SMALL_LAYERS,
            transformer=_torch_transformer_config(SMALL_D, SMALL_HIDDEN, SMALL_HEADS),
        ),
        use_variate_attention=True,
    )
    m.load_state_dict(_sd(w), strict=True)
    m.eval()
    _force_eps(m, PARITY_EPS)
    out, _, _ = m(_t(i["x"]), _t(i["patch_mask"]))
    return {"y": out.numpy()}


def _model_transformer_config() -> configs.TransformerConfig:
    return configs.TransformerConfig(
        model_dims=case_specs.MODEL_D,
        hidden_dims=case_specs.MODEL_HIDDEN,
        num_heads=case_specs.MODEL_HEADS,
        attention_norm="rms",
        feedforward_norm="rms",
        qk_norm="rms",
        use_bias=False,
        use_rope_seq=True,
        use_rope_var=False,
        ff_activation="relu",
        deterministic=True,
        use_memory_efficient_attention=True,
        use_sdpa=False,  # manual branch — mirrors the port (see module docstring)
    )


def _build_small_model(w) -> model.TimesFM3Torch:
    m = model.TimesFM3Torch(
        input_patch_len=case_specs.MODEL_PATCH_LEN,
        output_patch_len=case_specs.MODEL_OUT_LEN,
        quantiles=list(case_specs.MODEL_QUANTILES),
        residual_block_config=configs.ResidualBlockConfig(
            hidden_dims=case_specs.MODEL_D,
            output_dims=case_specs.MODEL_D,
            use_bias=False,
            activation="relu",
        ),
        transformer_config=configs.StackedTransformersConfig(
            num_layers=case_specs.MODEL_LAYERS,
            transformer=_model_transformer_config(),
        ),
    )
    m.load_state_dict(_sd(w), strict=True)
    m.eval()
    _force_eps(m, PARITY_EPS)
    return m


def _model_case(i, w, case_name: str) -> dict:
    m = _build_small_model(w)
    freeze_after = CASES[case_name].meta["freeze_after"]
    cpm = _t(i["patch_cpm_mask"]) if "patch_cpm_mask" in i else None
    out = m.forward(
        {
            "values": _t(i["values"]),
            "masks": _t(i["masks"]),
            "patch_is_target": _t(i["patch_is_target"]),
        },
        freeze_after=freeze_after,
        patch_cpm_mask=cpm,
        return_aux_outputs=True,
    )
    res = {
        "y": out["logits"].numpy(),
        "revin_mu": out["revin_stats"][0].numpy(),
        "revin_sigma": out["revin_stats"][1].numpy(),
        "aux_resblock_input": out["__call__:resblock_input"].numpy(),
        "aux_transformer_input": out["__call__:transformer_input"].numpy(),
        "aux_transformer_output": out["__call__:transformer_output"].numpy(),
    }
    for j, mask in enumerate(out["__call__:seq_attn_mask"]):
        res[f"seq_mask_{j}"] = _to_additive(mask)
    return res


def case_model_forward_small(i, w):
    return _model_case(i, w, "model_forward_small")


def case_model_forward_freeze(i, w):
    return _model_case(i, w, "model_forward_freeze")


CKPT_DIR = Path(__file__).resolve().parents[2] / "models" / "timesfm_3_0" / "original"


def _build_real_model() -> model.TimesFM3Torch:
    """Reference model at checkpoint config + REAL safetensors weights.

    Deliberate deviations from the untouched official setup (both recorded so
    the run cannot be misread as 'official default kernel parity', PLAN
    Phase-2 review note): use_sdpa is forced False (manual branch, mirroring
    the port) and eps is force-set to PARITY_EPS on both stacks (R5).
    """
    from safetensors.torch import load_file

    cfg = json.loads((CKPT_DIR / "config.json").read_text())
    tcfg = cfg.get("transformer_config", {}).get("transformer")
    if isinstance(tcfg, dict):
        tcfg["use_sdpa"] = False
    m = model.TimesFM3Torch(**cfg)
    m.load_state_dict(load_file(str(CKPT_DIR / "model.safetensors")), strict=True)
    m.eval()
    _force_eps(m, PARITY_EPS)
    return m


def _model_case_real(i) -> dict:
    m = _build_real_model()
    out = m.forward(
        {
            "values": _t(i["values"]),
            "masks": _t(i["masks"]),
            "patch_is_target": _t(i["patch_is_target"]),
        },
        freeze_after=None,
        patch_cpm_mask=_t(i["patch_cpm_mask"]),
        return_aux_outputs=True,
    )
    res = {
        "y": out["logits"].numpy(),
        "revin_mu": out["revin_stats"][0].numpy(),
        "revin_sigma": out["revin_stats"][1].numpy(),
        "aux_resblock_input": out["__call__:resblock_input"].numpy(),
        "aux_transformer_input": out["__call__:transformer_input"].numpy(),
        "aux_transformer_output": out["__call__:transformer_output"].numpy(),
    }
    for j, mask in enumerate(out["__call__:seq_attn_mask"]):
        res[f"seq_mask_{j}"] = _to_additive(mask)
    return res


def case_model_forward_real(i, w):
    return _model_case_real(i)


def case_stack_keys(i, w):
    """Dump the reference's 20-layer real-dim state-dict key tree (structure only)."""
    m = transformer.StackedMixingTransformer(
        configs.StackedTransformersConfig(
            num_layers=20, transformer=_torch_transformer_config(REAL_D, REAL_D, REAL_HEADS)
        ),
        use_variate_attention=True,
    )
    entries = sorted(
        f"{k}|{','.join(map(str, v.shape))}" for k, v in m.state_dict().items()
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
    "model_forward_small": case_model_forward_small,
    "model_forward_freeze": case_model_forward_freeze,
    "model_forward_real": case_model_forward_real,
}


def main(case: str, inputs_path: str, weights_path: str, outputs_path: str) -> None:
    inputs = _load_npz(inputs_path)
    weights = _load_npz(weights_path)
    out = CASE_RUNNERS[case](inputs, weights)

    meta = {
        "stack": "torch",
        "torch_version": torch.__version__,
        "torch_rmsnorm_eps_default": torch.nn.RMSNorm(80).eps,
        "effective_rmsnorm_eps": PARITY_EPS,
        "attention_path": "reference manual (use_sdpa=False, rescale_logits=False)",
        "device": "cpu",
    }
    np.savez(
        outputs_path,
        **{
            k: (v if isinstance(v, np.ndarray) else np.asarray(v.numpy()))
            for k, v in out.items()
        },
        _meta=np.array(json.dumps(meta)),
    )
    print(json.dumps({"op": case, "outputs": sorted(out), "meta": meta}))


if __name__ == "__main__":
    assert len(sys.argv) == 5, "usage: torch_runner.py <case> <in.npz> <w.npz|-> <out.npz>"
    assert Path(sys.argv[2]).exists(), sys.argv[2]
    main(*sys.argv[1:])
