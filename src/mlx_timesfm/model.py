"""TimesFM 3.0 model for Apple MLX (inference only).

Port of ``timesfm3.model.TimesFM3Torch`` — ``forward()`` (full-sequence),
``decode()`` (non-autoregressive single-pass with detrending/stitching) and the
thin user-facing ``forecast()``. The KV-cache decode path is not ported
(single-pass decode covers the inference API; see PLAN Phase 2 deferral).

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

import math
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
        self.use_stitching = config.use_stitching
        self.use_linear_detrending = config.use_linear_detrending
        self.linear_detrending_threshold = config.linear_detrending_threshold
        self.use_iterative_cpm_revin = config.use_iterative_cpm_revin
        self.use_frozen_running_stats = config.use_frozen_running_stats
        if config.use_stitching:
            self._stitching_extract_len = min(
                2 * config.input_patch_len, config.output_patch_len
            )

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

    # MLX convention: instances are called directly. (Reference only has
    # .forward; mlx.nn.Module does NOT auto-dispatch __call__ → forward, so
    # the alias must live on the class — instance attributes are not
    # consulted for special methods.)
    __call__ = forward

    # ------------------------------------------------------------------
    # decode (port of TimesFM3Torch.decode — non-autoregressive single pass)
    # ------------------------------------------------------------------

    def decode(
        self,
        target: mx.array,
        horizon: int = 0,
        past_only_covariates: mx.array | None = None,
        past_future_covariates: mx.array | None = None,
        target_mask: mx.array | None = None,
        past_only_mask: mx.array | None = None,
        past_future_mask: mx.array | None = None,
        mask: mx.array | None = None,
        return_aux_outputs: bool = False,
    ) -> mx.array | tuple[mx.array, dict[str, Any]]:
        """Port of ``TimesFM3Torch.decode`` — expression-for-expression.

        Args (all bool masks use True = masked/invalid):
          target: (b, u, context_len) float32.
          horizon: forecast horizon (inferred from past_future_covariates).
          past_only_covariates: (b, v_po, context_len) or None.
          past_future_covariates: (b, w, context_len + horizon) or None.
          target_mask / past_only_mask / past_future_mask: matching shapes.
          mask: (b, context_len) global mask or None.

        Returns logits (b, num_variates, horizon, num_quantiles).
        """
        batch_size, num_target, context = target.shape

        if past_future_covariates is not None:
            horizon = past_future_covariates.shape[-1] - context
        if horizon <= 0:
            raise ValueError("Decode function requires horizon > 0.")

        # 1. Left-pad context to a multiple of input_patch_len.
        ctx_padding = (self.input_patch_len - (context % self.input_patch_len)) % (
            self.input_patch_len
        )
        if ctx_padding > 0:
            target = _pad_left(target, ctx_padding, 0.0)
            if mask is not None:
                mask = _pad_left(mask, ctx_padding, True)
            if past_only_covariates is not None:
                # NOTE: reference pads the VALUES with 0 (no True fill) —
                # only masks get the True padding. Mirrored deliberately.
                past_only_covariates = _pad_left(past_only_covariates, ctx_padding, 0.0)
            if past_future_covariates is not None:
                past_future_covariates = _pad_left(
                    past_future_covariates, ctx_padding, 0.0
                )
            if target_mask is not None:
                target_mask = _pad_left(target_mask, ctx_padding, True)
            if past_only_mask is not None:
                past_only_mask = _pad_left(past_only_mask, ctx_padding, True)
            if past_future_mask is not None:
                past_future_mask = _pad_left(past_future_mask, ctx_padding, True)
            context += ctx_padding

        if mask is None:
            mask = _leading_true_mask(batch_size, context, ctx_padding)

        # 2. Horizon padding.
        if self.use_stitching:
            extract_len = self._stitching_extract_len
            overlap = extract_len - self.input_patch_len
            num_forecast_patches = max(
                math.ceil((horizon - overlap) / self.input_patch_len), 1
            )
            num_horizon_patches = num_forecast_patches + self.rolls - 1
            padded_horizon = num_horizon_patches * self.input_patch_len
            hor_padding = padded_horizon - horizon
        else:
            hor_padding = (-horizon) % self.output_patch_len
            padded_horizon = horizon + hor_padding
            num_horizon_patches = padded_horizon // self.input_patch_len
        num_context_patches = context // self.input_patch_len

        # 3. Context stream: target + past-only + past-future(context part).
        if target_mask is None:
            target_mask = mx.zeros(target.shape, dtype=mx.bool_)
        target_mask = target_mask | mask[:, None]

        all_ctx_vals = [target]
        all_ctx_masks = [target_mask]
        num_past_only = 0
        if past_only_covariates is not None:
            num_past_only = past_only_covariates.shape[1]
            if past_only_mask is None:
                past_only_mask = mx.zeros(
                    past_only_covariates.shape, dtype=mx.bool_
                )
            all_ctx_vals.append(past_only_covariates)
            all_ctx_masks.append(past_only_mask | mask[:, None])
        if past_future_covariates is not None:
            if past_future_mask is None:
                past_future_mask = mx.zeros(
                    past_future_covariates.shape, dtype=mx.bool_
                )
            all_ctx_vals.append(past_future_covariates[..., :context])
            all_ctx_masks.append(
                past_future_mask[..., :context] | mask[:, None]
            )

        ctx_vals = mx.concatenate(all_ctx_vals, axis=1)
        ctx_masks = mx.concatenate(all_ctx_masks, axis=1)

        # 4. Linear detrending (per variate, masked OLS on t/context).
        if self.use_linear_detrending:
            # torch.arange(-(context - 1), 1) → -(c-1) .. 0
            t_ctx = mx.arange(
                -(context - 1), 1, dtype=mx.float32
            )[None, None, :]
            t_n = t_ctx / context

            valid = ~ctx_masks
            vf = valid.astype(mx.float32)
            n_v = vf.sum(axis=-1, keepdims=True)
            sum_t = mx.where(valid, t_n, 0.0).sum(axis=-1, keepdims=True)
            sum_t2 = mx.where(valid, t_n * t_n, 0.0).sum(axis=-1, keepdims=True)
            sum_y = mx.where(valid, ctx_vals, 0.0).sum(axis=-1, keepdims=True)
            sum_ty = mx.where(valid, t_n * ctx_vals, 0.0).sum(
                axis=-1, keepdims=True
            )

            det = n_v * sum_t2 - sum_t * sum_t
            safe_det = mx.where(det == 0.0, mx.array(1.0), det)
            n_safe = mx.maximum(n_v, 1.0)
            m_trend = mx.where(
                det == 0.0, mx.array(0.0), (n_v * sum_ty - sum_t * sum_y) / safe_det
            )
            c_trend = mx.where(
                det == 0.0,
                mx.where(n_v > 0, sum_y / n_safe, mx.array(0.0)),
                (sum_y - m_trend * sum_t) / n_safe,
            )

            ctx_vals_detrended = ctx_vals - (m_trend * t_n + c_trend)

            mean_y = sum_y / n_safe
            sum_y2 = mx.where(valid, ctx_vals * ctx_vals, 0.0).sum(
                axis=-1, keepdims=True
            )
            var_orig = mx.maximum(sum_y2 / n_safe - mean_y * mean_y, 0.0)
            std_orig = mx.sqrt(var_orig)

            sum_yd = mx.where(valid, ctx_vals_detrended, 0.0).sum(
                axis=-1, keepdims=True
            )
            mean_yd = sum_yd / n_safe
            sum_yd2 = mx.where(
                valid, ctx_vals_detrended * ctx_vals_detrended, 0.0
            ).sum(axis=-1, keepdims=True)
            var_det = mx.maximum(sum_yd2 / n_safe - mean_yd * mean_yd, 0.0)
            std_det = mx.sqrt(var_det)

            apply_detrend = std_det < self.linear_detrending_threshold * std_orig
            ctx_vals = mx.where(apply_detrend, ctx_vals_detrended, ctx_vals)
        else:
            num_variates = ctx_vals.shape[1]
            z = mx.zeros((batch_size, num_variates, 1), dtype=mx.float32)
            m_trend, c_trend = z, z
            apply_detrend = mx.zeros((batch_size, num_variates, 1), dtype=mx.bool_)

        ctx_vals = mx.where(ctx_masks, 0.0, ctx_vals)

        # 5. Horizon stream.
        all_hor_vals = [
            mx.zeros((batch_size, num_target, padded_horizon), dtype=ctx_vals.dtype),
            mx.zeros(
                (batch_size, num_past_only, padded_horizon), dtype=ctx_vals.dtype
            ),
        ]
        all_hor_masks = [
            mx.ones((batch_size, num_target, padded_horizon), dtype=mx.bool_),
            mx.ones((batch_size, num_past_only, padded_horizon), dtype=mx.bool_),
        ]

        if past_future_covariates is not None:
            pf_future_vals = past_future_covariates[
                ..., context : context + horizon
            ]
            pf_future_masks = past_future_mask[..., context : context + horizon]
            if self.use_linear_detrending:
                off = num_target + num_past_only
                m_pf = m_trend[:, off:, :]
                c_pf = c_trend[:, off:, :]
                apply_detrend_pf = apply_detrend[:, off:, :]
                t_hor_pf = (
                    mx.arange(1, horizon + 1, dtype=mx.float32)[None, None, :]
                    / context
                )
                pf_trend_hor = m_pf * t_hor_pf + c_pf
                pf_future_vals = mx.where(
                    apply_detrend_pf, pf_future_vals - pf_trend_hor, pf_future_vals
                )
            pf_future_vals = mx.where(pf_future_masks, 0.0, pf_future_vals)
            if hor_padding > 0:
                pf_future_vals = _pad_right(pf_future_vals, hor_padding, 0.0)
                pf_future_masks = _pad_right(pf_future_masks, hor_padding, True)
            all_hor_vals.append(pf_future_vals)
            all_hor_masks.append(pf_future_masks)

        hor_vals = mx.concatenate(all_hor_vals, axis=1)
        hor_masks = mx.concatenate(all_hor_masks, axis=1)

        all_vals = mx.concatenate([ctx_vals, hor_vals], axis=-1)
        all_masks = mx.concatenate([ctx_masks, hor_masks], axis=-1)

        num_variates = all_vals.shape[1]
        total_patches = num_context_patches + num_horizon_patches
        patch_is_target = mx.concatenate(
            [
                mx.ones(
                    (batch_size, num_target + num_past_only, total_patches),
                    dtype=mx.bool_,
                ),
                mx.zeros(
                    (batch_size, num_variates - num_target - num_past_only,
                     total_patches),
                    dtype=mx.bool_,
                ),
            ],
            axis=1,
        )

        values_bvnp = all_vals.reshape(
            batch_size, num_variates, -1, self.input_patch_len
        )
        masks_bvnp = all_masks.reshape(
            batch_size, num_variates, -1, self.input_patch_len
        )

        inputs = {
            "values": values_bvnp,
            "masks": masks_bvnp,
            "patch_is_target": patch_is_target,
        }

        # Horizon CPM mask: context=False, horizon=True.
        horizon_cpm_mask = mx.concatenate(
            [
                mx.zeros(
                    (batch_size, num_context_patches),
                    dtype=mx.bool_,
                ),
                mx.ones((batch_size, num_horizon_patches), dtype=mx.bool_),
            ],
            axis=1,
        )

        freeze_after = (
            num_context_patches - 1 if self.use_frozen_running_stats else None
        )
        forward_out = self.forward(
            inputs,
            freeze_after=freeze_after,
            patch_cpm_mask=horizon_cpm_mask,
            return_aux_outputs=return_aux_outputs,
        )
        logits = forward_out["logits"]  # (b, v, n, output_patch_len, q)

        # 6. Forecast extraction.
        if self.use_stitching:
            forecast_indices = (
                mx.arange(num_forecast_patches) + (num_context_patches - 1)
            )
            patch_preds = mx.take(logits, forecast_indices, axis=2)[
                :, :, :, :extract_len, :
            ]
            horizon_logits = util.stitch_patches(
                patch_preds, self.input_patch_len
            )[:, :, :horizon, :]
        else:
            num_forecast_chunks = padded_horizon // self.output_patch_len
            forecast_indices = (
                mx.arange(num_forecast_chunks) * self.rolls
                + (num_context_patches - 1)
            )
            forecast_logits = mx.take(logits, forecast_indices, axis=2)
            horizon_logits = forecast_logits.reshape(
                batch_size, num_variates, -1, self.num_quantiles
            )[:, :, :horizon, :]

        # 7. Re-add the (thresholded) linear trend.
        if self.use_linear_detrending:
            t_forecast = (
                mx.arange(1, horizon + 1, dtype=mx.float32) / context
            )
            trend_forecast = (
                m_trend[:, :, 0, None] * t_forecast[None, None, :]
                + c_trend[:, :, 0, None]
            )
            trend_forecast = mx.where(
                apply_detrend[:, :, 0, None], trend_forecast, mx.array(0.0)
            )
            horizon_logits = horizon_logits + trend_forecast[:, :, :, None]

        if return_aux_outputs:
            return horizon_logits, forward_out
        return horizon_logits

    # ------------------------------------------------------------------
    # public forecasting API (SPEC F5)
    # ------------------------------------------------------------------

    def forecast(
        self,
        target: Any,
        horizon: int,
        past_only_covariates: Any = None,
        past_future_covariates: Any = None,
        mask: Any = None,
    ) -> mx.array:
        """Quantile forecasts for ``target``; returns (b, v, horizon, q).

        Input conventions (converted to float32 mx arrays):
          * ``target``: 1-D single series → (1, 1, T); 2-D ``(n_series, T)`` →
            batch of univariate series; 3-D (b, v, T) passed through.
          * covariates: 3-D (b, k, T) mx/numpy arrays (no reshaping magic).

        Thin wrapper over :meth:`decode` — series grouping, windowing and
        per-series max-context live in the (P2, unported) forecaster wrapper.
        """
        t = _as_float32(target, "target")
        if t.ndim == 1:
            t = t[None, None, :]
        elif t.ndim == 2:
            t = t[:, None, :]  # (n_series, T) → batch of univariate
        elif t.ndim != 3:
            raise ValueError(f"target must be 1-D/2-D/3-D, got shape {t.shape}")
        po = _as_float32(past_only_covariates, "past_only_covariates")
        pf = _as_float32(past_future_covariates, "past_future_covariates")
        m = None if mask is None else _as_float32(mask, "mask").astype(mx.bool_)
        logits = self.decode(
            t,
            horizon=horizon,
            past_only_covariates=po,
            past_future_covariates=pf,
            mask=m,
        )
        mx.eval(logits)
        return logits


# --------------------------------------------------------------------------
# decode helpers (torch-pad equivalents; MLX concatenation keeps dtype rules
# explicit and avoids mx.pad's constant_values edge cases on bool)
# --------------------------------------------------------------------------


def _pad_left(a: mx.array, n: int, value: Any) -> mx.array:
    fill = mx.full(a.shape[:-1] + (n,), value, dtype=a.dtype)
    return mx.concatenate([fill, a], axis=-1)


def _pad_right(a: mx.array, n: int, value: Any) -> mx.array:
    fill = mx.full(a.shape[:-1] + (n,), value, dtype=a.dtype)
    return mx.concatenate([a, fill], axis=-1)


def _leading_true_mask(b: int, length: int, num_true: int) -> mx.array:
    if num_true <= 0:
        return mx.zeros((b, length), dtype=mx.bool_)
    return mx.concatenate(
        [
            mx.ones((b, num_true), dtype=mx.bool_),
            mx.zeros((b, length - num_true), dtype=mx.bool_),
        ],
        axis=1,
    )


def _as_float32(x: Any, name: str) -> mx.array | None:
    if x is None:
        return None
    a = x if isinstance(x, mx.array) else mx.array(x)
    if a.dtype != mx.float32:
        a = a.astype(mx.float32)
    return a
