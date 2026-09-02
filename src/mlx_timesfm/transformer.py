# Copyright 2026 Google LLC
# Modifications Copyright 2026 AppAutomaton
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer layers for the MLX port.

Mirrors ``timesfm3.transformer`` (read-only PyTorch reference) module-for-module
and attribute-for-attribute, so checkpoint state-dict keys map 1:1 (SPEC R6):
``layers.{i}.seq_attn.query_proj.weight``, ``…seq_attn.per_dim_scale.per_dim_scale``,
``layers.{i}.pre_seq_attn_ln.weight``, ``ff0``/``ff1``, …

Mask convention (SPEC R2): the reference builds boolean masks, True = attend.
Our public ``make_attn_mask`` / ``make_segment_mask`` return the ADDITIVE float
form (attend = 0.0, block = −1e9 — the reference's manual-attention bias);
internally we combine booleans with the same expression order as the reference
and convert exactly once.

Attention path: manual (matmul + softmax with logits scaled by √head_dim via
pre-multiplying Q), which is exactly the reference's non-SDPA branch
(``use_sdpa=False, rescale_logits=False`` → net scale √d, MEA-equivalent).
``mx.fast.scaled_dot_product_attention`` handles head_dim=80 but differs from
manual by ~3e-3 on CPU (measured, 0.32.2) — above the A1 1e-4 gate, so the fast
kernel stays a Phase-6 performance option under A2 tolerances, not the default.

RoPE is custom (SPEC R3): half-rotation, 3D/4D inputs, arbitrary (b, n) integer
positions; ``mlx.nn.RoPE`` is NOT used (int-offset-only API).
"""

from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn

from .config import TransformerConfig
from .normalization import PerDimScale, RMSNorm
from .util import get_activation

__all__ = [
    "rope",
    "make_attn_mask",
    "make_segment_mask",
    "MultiHeadAttention",
    "MixingTransformer",
    "StackedMixingTransformer",
]

_MASK_VALUE = -1e9  # reference manual-path bias (exact in fp32)


# --------------------------------------------------------------------------
# RoPE
# --------------------------------------------------------------------------


def rope(
    inputs: mx.array,
    position: mx.array | None = None,
    *,
    embedding_dims: int | None = None,
    min_timescale: float = 1.0,
    max_timescale: float = 10000.0,
) -> mx.array:
    """Rotary embeddings, port of RotaryPositionalEmbedding.forward (half-rotation).

    Args:
        inputs: (b, n, d) or (b, n, h, hd).
        position: (b, n) ints/floats; None → arange(n) broadcast over batch.
    """
    d = embedding_dims if embedding_dims is not None else inputs.shape[-1]
    if d != inputs.shape[-1]:
        raise ValueError(
            "The embedding dims of the rotary position embedding "
            "must match the hidden dimension of the inputs."
        )
    half = d // 2
    fraction = 2.0 * mx.arange(half, dtype=mx.float32) / d
    timescale = min_timescale * (max_timescale / min_timescale) ** fraction

    if position is None:
        seq_length = inputs.shape[1]
        position = mx.arange(seq_length, dtype=mx.float32).reshape(1, seq_length)

    if inputs.ndim == 4:
        pos = position[..., None, None]
        ts = timescale.reshape(1, 1, 1, half)
    elif inputs.ndim == 3:
        pos = position[..., None]
        ts = timescale.reshape(1, 1, half)
    else:
        raise ValueError("Inputs must be of rank 3 or 4.")

    sinusoid_inp = pos.astype(mx.float32) / ts
    sin_val = mx.sin(sinusoid_inp)
    cos_val = mx.cos(sinusoid_inp)
    first_half, second_half = mx.split(inputs, 2, axis=-1)
    first_part = first_half * cos_val - second_half * sin_val
    second_part = second_half * cos_val + first_half * sin_val
    return mx.concatenate([first_part, second_part], axis=-1)


# --------------------------------------------------------------------------
# Masks (public: additive float. attend = 0.0, block = _MASK_VALUE)
# --------------------------------------------------------------------------


def _attn_mask_bool(
    query_length: int,
    num_all_masked_kv: mx.array,
    query_index_offset: mx.array | None = None,
    kv_length: int = 0,
    causal: bool = True,
) -> mx.array:
    """True = attend. Mirrors reference make_attn_mask (b, 1, q, kv)."""
    if kv_length == 0:
        kv_length = query_length
    q_index = mx.arange(query_length).reshape(1, 1, query_length, 1)
    if query_index_offset is not None:
        q_index = q_index + query_index_offset.reshape(-1, 1, 1, 1)
    kv_index = mx.arange(kv_length).reshape(1, 1, 1, kv_length)
    mask = kv_index >= num_all_masked_kv.reshape(-1, 1, 1, 1)
    if causal:
        return (q_index >= kv_index) & mask
    return mask


def _segment_mask_bool(segment_ids: mx.array) -> mx.array:
    """True = same segment. Mirrors reference make_segment_mask (b, 1, n, n)."""
    return (segment_ids[:, :, None] == segment_ids[:, None, :])[:, None, :, :]


def _to_additive(mask: mx.array) -> mx.array:
    return mx.where(mask, 0.0, _MASK_VALUE).astype(mx.float32)


def make_attn_mask(
    query_length: int,
    num_all_masked_kv: mx.array,
    query_index_offset: mx.array | None = None,
    kv_length: int = 0,
    causal: bool = True,
) -> mx.array:
    """Additive attention mask: attend = 0.0, block = −1e9. Shape (b, 1, q, kv)."""
    return _to_additive(
        _attn_mask_bool(
            query_length, num_all_masked_kv, query_index_offset, kv_length, causal
        )
    )


def make_segment_mask(segment_ids: mx.array) -> mx.array:
    """Additive segment mask from ids (b, n) → (b, 1, n, n); cross-segment = −1e9."""
    return _to_additive(_segment_mask_bool(segment_ids))


# --------------------------------------------------------------------------
# Attention
# --------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """MHA: proj → RoPE → QK-RMSNorm → PerDimScale(Q) → attention(scale=√head_dim) → out_proj.

    Mirrors the reference class attribute names (no-bias projections,
    ``per_dim_scale.per_dim_scale``). KV-cache is out of scope here (Phase 4).
    Attention kernel: manual path — net logit scale is √head_dim, achieved the
    same way as the reference manual branch (Q pre-multiplied by √head_dim).
    """

    def __init__(
        self,
        num_heads: int,
        in_features: int,
        *,
        eps: float,
        use_per_dim_scale: bool = True,
        use_rotary_position_embeddings: bool = True,
        causal_attention: bool = True,
        use_bias: bool = False,
        qk_norm: str = "rms",
        v_norm: str = "none",
    ):
        super().__init__()
        if in_features % num_heads:
            raise ValueError("in_features must divide into num_heads")
        self.num_heads = num_heads
        self.in_features = in_features
        self.head_dim = in_features // num_heads
        self.causal_attention = causal_attention
        self.use_rope = use_rotary_position_embeddings  # no params: safe name

        self.query_proj = nn.Linear(in_features, in_features, bias=use_bias)
        self.key_proj = nn.Linear(in_features, in_features, bias=use_bias)
        self.value_proj = nn.Linear(in_features, in_features, bias=use_bias)
        self.out_proj = nn.Linear(in_features, in_features, bias=use_bias)

        self.query_ln = RMSNorm(self.head_dim, eps) if qk_norm == "rms" else None
        self.key_ln = RMSNorm(self.head_dim, eps) if qk_norm == "rms" else None
        self.value_ln = (
            RMSNorm(self.head_dim, eps, affine=False) if v_norm == "rms" else None
        )
        self.per_dim_scale = PerDimScale(self.head_dim) if use_per_dim_scale else None

    def __call__(
        self,
        inputs_q: mx.array,
        *,
        segment_ids: mx.array | None = None,
        segment_pos: mx.array | None = None,
        patch_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """(b, n, d) → (output (b, n, d), additive attn mask (b, 1, n, n))."""
        b, n, _ = inputs_q.shape
        h, hd = self.num_heads, self.head_dim
        if patch_mask is None:
            patch_mask = mx.zeros((b, n), dtype=mx.bool_)

        query = self.query_proj(inputs_q).reshape(b, n, h, hd)
        key = self.key_proj(inputs_q).reshape(b, n, h, hd)
        value = self.value_proj(inputs_q).reshape(b, n, h, hd)

        if self.use_rope:
            position = (
                segment_pos
                if segment_pos is not None
                else mx.arange(n, dtype=mx.int32).reshape(1, n)
            )
            query = rope(query, position, embedding_dims=hd)
            key = rope(key, position, embedding_dims=hd)

        if self.query_ln is not None:
            query = self.query_ln(query)
        if self.key_ln is not None:
            key = self.key_ln(key)
        if self.per_dim_scale is not None:
            query = self.per_dim_scale(query)
        if self.value_ln is not None:
            value = self.value_ln(value)

        # Mask combination, same expression as the reference full-sequence path:
        # causal ⊧ kv-visible AND NOT patch_mask (AND segment mask if given).
        attn = ~patch_mask[:, None, None, :]
        if self.causal_attention:
            q_index = mx.arange(n).reshape(1, 1, n, 1)
            kv_index = mx.arange(n).reshape(1, 1, 1, n)
            attn = (q_index >= kv_index) & attn
        if segment_ids is not None:
            attn = attn & _segment_mask_bool(segment_ids)
        # Shape mirrors the reference: (b,1,n,n) causal; (b,1,1,n) non-causal
        # (reference does not broadcast). Broadcasts against logits anyway.
        attn_bias = _to_additive(attn)

        query = query.transpose(0, 2, 1, 3)  # (b, h, n, hd)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        # Manual path, mirroring the reference: fold √head_dim into Q, then
        # logits = QKᵀ + bias. Net logit scale = √head_dim (R1).
        query = query * math.sqrt(hd)
        attn_logits = query @ key.swapaxes(-1, -2) + attn_bias
        x = mx.softmax(attn_logits, axis=-1) @ value

        x = x.transpose(0, 2, 1, 3).reshape(b, n, self.in_features)
        return self.out_proj(x), attn_bias


# --------------------------------------------------------------------------
# Mixing transformer
# --------------------------------------------------------------------------


class MixingTransformer(nn.Module):
    """One layer: seq-attn → var-attn → FFN; residual is ALWAYS ``post_ln(sublayer_out) + x``.

    Reference parity note: the residual is post_ln(attn_out) + x, NOT
    norm(x + attn_out). Attribute order matches the reference state dict.
    """

    def __init__(
        self,
        config: TransformerConfig,
        *,
        eps: float,
        use_variate_attention: bool = True,
    ):
        super().__init__()
        self.config = config
        self.use_variate_attention = use_variate_attention

        self.pre_seq_attn_ln = RMSNorm(config.model_dims, eps)
        self.post_seq_attn_ln = RMSNorm(config.model_dims, eps)
        self.seq_attn = MultiHeadAttention(
            num_heads=config.num_heads,
            in_features=config.model_dims,
            eps=eps,
            use_per_dim_scale=True,
            use_rotary_position_embeddings=config.use_rope_seq,
            causal_attention=config.causal_attention,
            use_bias=config.use_bias,
            qk_norm=config.qk_norm,
            v_norm=config.v_norm,
        )

        if use_variate_attention:
            self.pre_var_attn_ln = RMSNorm(config.model_dims, eps)
            self.post_var_attn_ln = RMSNorm(config.model_dims, eps)
            self.var_attn = MultiHeadAttention(
                num_heads=config.num_heads,
                in_features=config.model_dims,
                eps=eps,
                use_per_dim_scale=True,
                use_rotary_position_embeddings=config.use_rope_var,
                causal_attention=False,
                use_bias=config.use_bias,
                qk_norm=config.qk_norm,
                v_norm=config.v_norm,
            )

        self.pre_ff_ln = RMSNorm(config.model_dims, eps)
        self.post_ff_ln = RMSNorm(config.model_dims, eps)
        self.ff0 = nn.Linear(config.model_dims, config.hidden_dims, bias=config.use_bias)
        self.ff1 = nn.Linear(config.hidden_dims, config.model_dims, bias=config.use_bias)
        self.activation = get_activation(config.ff_activation)

    def __call__(
        self,
        input_embeddings: mx.array,
        patch_mask: mx.array,
        segment_ids: mx.array | None = None,
        segment_pos: mx.array | None = None,
        var_segment_pos: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """(b, v, n, d) → (output (b, v, n, d), seq-attn additive mask)."""
        b, v, n, d = input_embeddings.shape

        # --- Sequence attention ---
        seq_in = self.pre_seq_attn_ln(input_embeddings).reshape(b * v, n, d)
        patch_flat = patch_mask.reshape(b * v, n)
        seg_ids_flat = (
            mx.broadcast_to(segment_ids[:, None, :], (b, v, n)).reshape(b * v, n)
            if segment_ids is not None
            else None
        )
        seg_pos_flat = (
            mx.broadcast_to(segment_pos[:, None, :], (b, v, n)).reshape(b * v, n)
            if segment_pos is not None
            else None
        )
        seq_out, seq_mask = self.seq_attn(
            seq_in,
            segment_ids=seg_ids_flat,
            segment_pos=seg_pos_flat,
            patch_mask=patch_flat,
        )
        h1 = self.post_seq_attn_ln(seq_out.reshape(b, v, n, d)) + input_embeddings

        # --- Variate attention ---
        if self.use_variate_attention:
            var_in = self.pre_var_attn_ln(h1).transpose(0, 2, 1, 3).reshape(b * n, v, d)
            var_patch = patch_mask.transpose(0, 2, 1).reshape(b * n, v)
            var_out, _ = self.var_attn(
                var_in, segment_pos=var_segment_pos, patch_mask=var_patch
            )
            var_out = var_out.reshape(b, n, v, d).transpose(0, 2, 1, 3)
            h2 = self.post_var_attn_ln(var_out) + h1
        else:
            h2 = h1

        # --- Feed forward ---
        ff_out = self.ff1(self.activation(self.ff0(self.pre_ff_ln(h2))))
        return self.post_ff_ln(ff_out) + h2, seq_mask


class StackedMixingTransformer(nn.Module):
    """``config.num_layers`` MixingTransformers; attribute tree = ``layers.{i}.…``."""

    def __init__(
        self,
        num_layers: int,
        config: TransformerConfig,
        *,
        eps: float,
        use_variate_attention: bool = True,
    ):
        super().__init__()
        self.layers = [
            MixingTransformer(
                config, eps=eps, use_variate_attention=use_variate_attention
            )
            for _ in range(num_layers)
        ]

    def __call__(
        self,
        input_embeddings: mx.array,
        patch_mask: mx.array,
        segment_ids: mx.array | None = None,
        segment_pos: mx.array | None = None,
        var_segment_pos: mx.array | None = None,
    ) -> tuple[mx.array, list[mx.array]]:
        out = input_embeddings
        masks = []
        for layer in self.layers:
            out, layer_mask = layer(
                out, patch_mask, segment_ids, segment_pos, var_segment_pos
            )
            masks.append(layer_mask)
        return out, masks
