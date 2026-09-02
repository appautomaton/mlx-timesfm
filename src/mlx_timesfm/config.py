"""TimesFM 3.0 configuration dataclasses.

Mirrors the semantics of ``timesfm3.configs`` (read-only PyTorch reference) plus
the top-level fields of ``TimesFM3Torch.__init__`` / ``to_dict`` — which is
exactly the shape of ``config.json`` shipped with the checkpoint.

This is the single source of truth for model construction, tests and
``mlx_timesfm.load()``.

RMSNorm epsilon (SPEC R5): ``TimesFM3Config.rmsnorm_eps`` is a *required knob*,
never a baked-in default. The reference checkpoint's config.json does not carry
an eps value, and both frameworks' RMSNorm defaults are version-dependent —
so model construction must receive an explicit value (the parity harness probes
the reference torch env's actual eps and records it in every report).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "INFERENCE_RMSNORM_EPS",
    "ResidualBlockConfig",
    "TransformerConfig",
    "StackedTransformersConfig",
    "TimesFM3Config",
    "load_config",
]

# Documented port-level RMSNorm eps for INFERENCE (applied by load_config /
# mlx_timesfm.load). SPEC R5 forbids *guessing* a framework's RMSNorm default;
# this is an explicit, stable port choice, written down and test-locked.
# Parity code never uses it — it passes the reference env's probed effective
# value instead (torch 2.13 reports `.eps is None`; effective behavior is
# settled at 1b).
INFERENCE_RMSNORM_EPS = 1e-5


@dataclasses.dataclass(frozen=True)
class ResidualBlockConfig:
    """Framework-agnostic config for a residual block."""

    hidden_dims: int
    output_dims: int
    use_bias: bool
    activation: Literal["relu", "swish", "none"]
    dropout: float = 0.0
    identity_skip: bool = False
    prenorm: Literal["rms", "none"] = "none"


@dataclasses.dataclass(frozen=True)
class TransformerConfig:
    """Framework-agnostic config for a transformer layer."""

    model_dims: int
    hidden_dims: int
    num_heads: int
    attention_norm: Literal["rms"]
    feedforward_norm: Literal["rms"]
    qk_norm: Literal["rms", "none"]
    use_bias: bool
    use_rope_seq: bool
    use_rope_var: bool
    ff_activation: Literal["relu", "swish", "none", "swiglu"]
    deterministic: bool
    v_norm: Literal["rms", "none"] = "none"
    causal_attention: bool = True
    debug_no_masking: bool = False
    training: bool = True
    use_memory_efficient_attention: bool = True
    paired_token_skip_second: bool = False
    max_variates: int = 32
    # PyTorch-only: when True uses F.scaled_dot_product_attention. In the MLX
    # port this selects mx.fast.scaled_dot_product_attention (always taken);
    # kept so config.json round-trips 1:1 with the reference.
    use_sdpa: bool = True


@dataclasses.dataclass(frozen=True)
class StackedTransformersConfig:
    """Framework-agnostic config for the transformer stack."""

    num_layers: int
    transformer: TransformerConfig
    use_remat: bool = True  # training-only in the reference; inert here


@dataclasses.dataclass(frozen=True)
class TimesFM3Config:
    """Top-level model config; field-for-field what config.json contains."""

    input_patch_len: int = 32
    output_patch_len: int = 64
    quantiles: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    residual_block_config: ResidualBlockConfig = dataclasses.field(
        default_factory=lambda: ResidualBlockConfig(
            hidden_dims=1280, output_dims=1280, use_bias=False, activation="relu"
        )
    )
    transformer_config: StackedTransformersConfig = dataclasses.field(
        default_factory=lambda: StackedTransformersConfig(
            num_layers=20,
            transformer=TransformerConfig(
                model_dims=1280,
                hidden_dims=1280,
                num_heads=16,
                attention_norm="rms",
                feedforward_norm="rms",
                qk_norm="rms",
                use_bias=False,
                use_rope_seq=True,
                use_rope_var=False,
                ff_activation="relu",
                deterministic=True,
            ),
        )
    )
    use_variate_attention: bool = True
    value_clip: float = 1e20
    use_stitching: bool = True
    use_linear_detrending: bool = True
    linear_detrending_threshold: float = 0.5
    use_iterative_cpm_revin: bool = True
    use_frozen_running_stats: bool = False
    input_transform: str = "identity"
    # Port-only knob, excluded from to_dict() (checkpoint json has no such
    # field). None means "unset" — from_dict never invents a value (R5).
    # load_config applies the documented INFERENCE_RMSNORM_EPS at the
    # inference boundary; parity overrides with the probed reference value.
    rmsnorm_eps: float | None = None

    def __post_init__(self) -> None:
        # Same validation as TimesFM3Torch.__init__ (reference model.py).
        object.__setattr__(self, "quantiles", tuple(self.quantiles))
        if self.output_patch_len % self.input_patch_len != 0:
            raise ValueError(
                f"Output patch len {self.output_patch_len} must be a multiple of"
                f" input patch len {self.input_patch_len}."
            )
        if self.residual_block_config.output_dims != self.transformer_config.transformer.model_dims:
            raise ValueError(
                "ResidualBlock output_dims must match Transformer model_dims."
            )
        if self.use_stitching and self.output_patch_len <= self.input_patch_len:
            raise ValueError("use_stitching requires output_patch_len > input_patch_len")

    @property
    def num_quantiles(self) -> int:
        return len(self.quantiles)

    @property
    def rolls(self) -> int:
        return self.output_patch_len // self.input_patch_len

    @property
    def resblock_input_dim(self) -> int:
        """Eager replacement for the torch lazy ``set_input_dims`` hack."""
        return 2 * (self.input_patch_len + self.output_patch_len)

    def to_dict(self) -> dict[str, Any]:
        """Checkpoint-shaped dict — matches the official config.json key set
        exactly, so ``to_dict`` ↔ ``from_dict`` round-trips against it.

        Port-only knobs (``rmsnorm_eps``) are deliberately excluded: the
        checkpoint json has no such field.
        """
        d = dataclasses.asdict(self)
        d["quantiles"] = list(self.quantiles)
        d.pop("rmsnorm_eps", None)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimesFM3Config:
        """Build from a raw config dict (e.g. parsed config.json).

        Nested dicts become dataclasses; unknown keys raise loudly rather than
        being silently dropped (R6 philosophy).
        """
        kwargs = dict(data)
        rb = kwargs.get("residual_block_config")
        if isinstance(rb, dict):
            kwargs["residual_block_config"] = ResidualBlockConfig(**rb)
        tc = kwargs.get("transformer_config")
        if isinstance(tc, dict):
            t = tc.get("transformer")
            if isinstance(t, dict):
                tc = {**tc, "transformer": TransformerConfig(**t)}
            kwargs["transformer_config"] = StackedTransformersConfig(**tc)
        if "quantiles" in kwargs:
            kwargs["quantiles"] = tuple(kwargs["quantiles"])
        return cls(**kwargs)


def load_config(path: str | Path, rmsnorm_eps: float | None = None) -> TimesFM3Config:
    """Load config.json from a model directory (or a direct path to the json).

    This is the *inference* boundary (what ``mlx_timesfm.load()`` will call):
    checkpoint configs carry no eps, so we apply the documented
    ``INFERENCE_RMSNORM_EPS`` unless a value is passed explicitly. Parity code
    passes the reference env's probed effective eps instead (SPEC R5).
    ``from_dict`` stays faithful — it never invents an eps.
    """
    p = Path(path)
    if p.is_dir():
        p = p / "config.json"
    with open(p) as f:
        cfg = TimesFM3Config.from_dict(json.load(f))
    if cfg.rmsnorm_eps is None:
        cfg = dataclasses.replace(
            cfg, rmsnorm_eps=rmsnorm_eps if rmsnorm_eps is not None else INFERENCE_RMSNORM_EPS
        )
    return cfg
