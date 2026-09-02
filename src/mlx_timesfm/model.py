"""TimesFM 3.0 model for Apple MLX (inference only).

Port of ``timesfm3.model.TimesFM3Torch`` — ``forward()`` (full-sequence) and
``_preprocess`` only. ``decode()`` / KV-cache come in Phase 4; this module
raises if asked for them implicitly (no cache parameters exist).

State-dict names mirror the reference exactly (SPEC R6, verified against the
checkpoint's 445 tensors): ``pre_transformer_resblock.{hidden,output,residual}_
layer.weight``, ``transformer_stack.layers.{i}.*``, ``output_head.weight`` +
``output_head.bias`` (the ONLY biased linear; CPM/quantile head).

Key reference behaviours deliberately preserved:
  * running stats are computed BEFORE the CPM mask is folded into ``masks``
    (CPM only affects the revin-refine path + covariate masking);
  * ``forward`` masks only *leading* fully-masked patches:
    ``effective = cumprod(patch_mask, axis=2)`` — horizon patches stay
    visible to attention (SPEC §7 watch-out);
  * reverse RevIN then clamp(±value_clip) then reshape to
    (b, v, n, output_patch_len, num_quantiles).
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx import nn

from . import util
from .config import TimesFM3Config
from .cpm_revin_refine import cpm_iterative_revin_refine
from .dense import ResidualBlock
from .transformer import StackedMixingTransformer

__all__ = ["TimesFM3"]


class TimesFM3(nn.Module):
    """Inference-only TimesFM 3.0 (full-sequence forward)."""

    def __init__(self, config: TimesFM3Config):
        super().__init__()
        if config.rmsnorm_eps is None:
            raise ValueError(
                "TimesFM3Config.rmsnorm_eps is unset — build configs through "
                "load_config() (applies INFERENCE_RMSNORM_EPS) or pass an "
                "explicit eps (SPEC R5: never inherit a framework default)."
            )
        eps = config.rmsnorm_eps
        tc = config.transformer_config

        self.config = config
        self.input_patch_len = config.input_patch_len
        self.output_patch_len = config.output_patch_len
        self.num_quantiles = config.num_quantiles
        self.rolls = config.rolls
        self.value_clip = config.value_clip
        self.use_iterative_cpm_revin = config.use_iterative_cpm_revin

        self.pre_transformer_resblock = ResidualBlock(
            config.residual_block_config,
            input_dim=config.resblock_input_dim,  # eager; no set_input_dims
            eps=eps,
        )
        self.transformer_stack = StackedMixingTransformer(
            tc.num_layers,
            tc.transformer,
            eps=eps,
            use_variate_attention=config.use_variate_attention,
        )
        self.output_head = nn.Linear(
            tc.transformer.model_dims,
            config.output_patch_len * config.num_quantiles,
            bias=True,  # the one biased projection (checkpoint output_head.bias)
        )

    # ------------------------------------------------------------------
    # preprocessing (RevIN + covariate rolls + ResBlock) — port of _preprocess
    # ------------------------------------------------------------------

    def _preprocess(
        self,
        values: mx.array,
        masks: mx.array,
        patch_is_target: mx.array,
        freeze_after: int | None = None,
        patch_cpm_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, tuple[mx.array, mx.array], mx.array]:
        running_n, running_mean, running_std = util.get_running_stats(values, masks)
        if freeze_after is not None:
            n = values.shape[2]
            if 0 <= freeze_after < n - 1:
                # reference mutates in place; we rebuild functionally
                def _freeze(a: mx.array) -> mx.array:
                    frozen = mx.broadcast_to(
                        a[:, :, freeze_after : freeze_after + 1],
                        (a.shape[0], a.shape[1], n - freeze_after - 1),
                    )
                    return mx.concatenate(
                        [a[:, :, : freeze_after + 1], frozen], axis=2
                    )

                running_mean = _freeze(running_mean)
                running_std = _freeze(running_std)

        # CPM mask: additionally mask TARGET variates at CPM patch positions.
        if patch_cpm_mask is not None:
            cpm_bvnp = patch_cpm_mask[:, None, :, None]
            masks = masks | (cpm_bvnp & patch_is_target[..., None])

        values_bvnp = util.revin(values, running_mean, running_std, reverse=False)
        values_bvnp = mx.where(masks, 0.0, values_bvnp)

        # Future-covariate patches: rolled values + rolled masks.
        values_fcov, wrap_mask = util.get_output_patch_via_roll(values, self.rolls)
        values_fcov = util.revin(values_fcov, running_mean, running_std, reverse=False)
        masks_fcov_raw, _ = util.get_output_patch_via_roll(masks, self.rolls)
        masks_fcov = masks_fcov_raw | patch_is_target[..., None] | wrap_mask
        values_fcov = mx.where(masks_fcov, 0.0, values_fcov)

        values_cat = mx.concatenate([values_bvnp, values_fcov], axis=-1)
        masks_cat = mx.concatenate([masks, masks_fcov], axis=-1)

        resblock_input = mx.concatenate(
            [values_cat, masks_cat.astype(mx.float32)], axis=-1
        )
        resblock_output = self.pre_transformer_resblock(resblock_input)

        # patch fully masked iff ALL its points are masked
        patch_mask_bvn = masks_cat.all(axis=3)
        return (
            resblock_input,
            resblock_output,
            patch_mask_bvn,
            (running_mean, running_std),
            running_n,
        )

    # ------------------------------------------------------------------
    # forward (port of TimesFM3Torch.forward)
    # ------------------------------------------------------------------

    def forward(
        self,
        inputs: dict[str, Any],
        freeze_after: int | None = None,
        patch_cpm_mask: mx.array | None = None,
        return_aux_outputs: bool = False,
    ) -> dict[str, Any]:
        values = mx.nan_to_num(inputs["values"], nan=0.0)
        values = mx.clip(values, -self.value_clip, self.value_clip)
        masks = inputs["masks"].astype(mx.bool_)
        patch_is_target = inputs["patch_is_target"]

        p = values.shape[3]
        if p != self.input_patch_len:
            raise ValueError(
                f"Input patch_len {p} != model input_patch_len {self.input_patch_len}"
            )

        (
            resblock_input,
            transformer_input,
            transformer_patch_mask,
            revin_stats,
            running_n,
        ) = self._preprocess(
            values,
            masks,
            patch_is_target,
            freeze_after=freeze_after,
            patch_cpm_mask=patch_cpm_mask,
        )

        # Only LEADING fully-masked patches are masked for attention
        # (cumprod semantics; horizon patches stay visible).
        effective_patch_mask = (
            mx.cumprod(transformer_patch_mask.astype(mx.float32), axis=2) > 0.5
        )
        transformer_output, seq_attn_masks = self.transformer_stack(
            transformer_input, effective_patch_mask
        )

        raw_logits = self.output_head(transformer_output)
        revin_mean, revin_std = revin_stats

        if self.use_iterative_cpm_revin and patch_cpm_mask is not None:
            refined_mu, refined_sigma = cpm_iterative_revin_refine(
                raw_logits,
                revin_n=running_n,
                revin_mu=revin_mean,
                revin_sigma=revin_std,
                patch_cpm_mask=patch_cpm_mask,
                median_q_idx=self.num_quantiles // 2,
                rolls=self.rolls,
                patch_len=self.input_patch_len,
                num_quantiles=self.num_quantiles,
                value_clip=self.value_clip,
            )
            cpm_bvn = patch_cpm_mask[:, None, :]
            revin_mean = mx.where(cpm_bvn, refined_mu, revin_mean)
            revin_std = mx.where(cpm_bvn, refined_sigma, revin_std)

        revin_logits = util.revin(raw_logits, revin_mean, revin_std, reverse=True)
        clipped_logits = mx.clip(revin_logits, -self.value_clip, self.value_clip)

        b, v, n_patches = clipped_logits.shape[:3]
        final_logits = clipped_logits.reshape(
            b, v, n_patches, self.output_patch_len, self.num_quantiles
        )

        # NOTE: reference returns the ORIGINAL _preprocess stats tuple here,
        # not the cpm-refined revin_mean/std (those only affect logits).
        outputs: dict[str, Any] = {"logits": final_logits, "revin_stats": revin_stats}
        if return_aux_outputs:
            outputs["__call__:resblock_input"] = resblock_input
            outputs["__call__:transformer_input"] = transformer_input
            outputs["__call__:seq_attn_mask"] = seq_attn_masks
            outputs["__call__:transformer_output"] = transformer_output
        return outputs
