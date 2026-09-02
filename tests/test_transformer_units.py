"""Property tests for the ported ops — torch-free (run everywhere)."""

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_timesfm.config import TransformerConfig
from mlx_timesfm.normalization import PerDimScale, RMSNorm
from mlx_timesfm.transformer import (
    MixingTransformer,
    MultiHeadAttention,
    make_attn_mask,
    make_segment_mask,
    rope,
)


def _np(x: mx.array) -> np.ndarray:
    mx.eval(x)
    return np.asarray(x)


# ---- RMSNorm ----


def test_rmsnorm_requires_explicit_eps() -> None:
    with pytest.raises(ValueError, match="explicitly"):
        RMSNorm(80)
    assert RMSNorm(80, eps=1e-5).eps == 1e-5


def test_rmsnorm_matches_naive_numpy() -> None:
    x = mx.array(np.random.default_rng(0).normal(size=(3, 7, 80)).astype(np.float32))
    eps = 1e-5
    m = RMSNorm(80, eps=eps)
    m.weight = mx.array(np.random.default_rng(1).normal(1, 0.05, (80,)).astype(np.float32))
    got = _np(m(x))
    xn = np.asarray(x)
    want = xn / np.sqrt((xn**2).mean(-1, keepdims=True) + eps) * np.asarray(m.weight)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


# ---- PerDimScale ----


def test_per_dim_scale_zero_init_is_reciprocal_sqrt_d() -> None:
    # softplus(0) = ln 2; RECIPROCAL_OF_SOFTPLUS_0 * ln2 ≈ 1 ⇒ net scale ≈ 1/√d.
    d = 80
    m = PerDimScale(d)
    ones = mx.ones((1, d))
    scaled = _np(m(ones))
    np.testing.assert_allclose(scaled, np.full((1, d), 1 / math.sqrt(d)), rtol=1e-5)


# ---- RoPE ----


def test_rope_preserves_l2_norm() -> None:
    rng = np.random.default_rng(0)
    x = mx.array(rng.normal(size=(2, 12, 4, 80)).astype(np.float32))
    pos = mx.array(rng.integers(0, 1000, (2, 12)))
    before = np.linalg.norm(np.asarray(x), axis=-1)
    after = np.linalg.norm(_np(rope(x, pos)), axis=-1)
    np.testing.assert_allclose(after, before, rtol=1e-4)


def test_rope_zero_position_is_identity() -> None:
    x = mx.array(np.random.default_rng(1).normal(size=(1, 5, 80)).astype(np.float32))
    zero_pos = mx.zeros((1, 5), dtype=mx.int32)
    np.testing.assert_allclose(_np(rope(x, zero_pos)), np.asarray(x), rtol=1e-6, atol=1e-6)


def test_rope_accepts_per_batch_positions_and_none_default() -> None:
    x = mx.array(np.random.default_rng(2).normal(size=(2, 6, 4, 80)).astype(np.float32))
    # None → arange(n), broadcast over batch
    got_none = _np(rope(x, None))
    got_arange = _np(rope(x, mx.arange(6).reshape(1, 6)))
    np.testing.assert_allclose(got_none, got_arange, rtol=0, atol=0)
    # per-batch different positions must differ between batch rows
    pos = mx.stack([mx.arange(6), mx.arange(6) + 13])
    out = _np(rope(x, pos))
    assert not np.allclose(out[0], out[1])


def test_rope_rejects_rank_mismatch() -> None:
    x = mx.zeros((2, 3))
    with pytest.raises(ValueError):
        rope(x, None, embedding_dims=80)
    with pytest.raises(ValueError):
        rope(mx.zeros((2, 3, 4, 5, 6)), None)


# ---- masks ----


def test_make_attn_mask_causal_exact_against_numpy() -> None:
    num_masked = np.array([0, 2], dtype=np.int32)
    offset = np.array([0, 3], dtype=np.int32)
    got = _np(
        make_attn_mask(5, mx.array(num_masked), mx.array(offset), kv_length=7, causal=True)
    )
    q = np.arange(5)[None, None, :, None] + offset[:, None, None, None]
    kv = np.arange(7)[None, None, None, :]
    allow = (q >= kv) & (kv >= num_masked[:, None, None, None])
    want = np.where(allow, 0.0, -1e9).astype(np.float32)
    np.testing.assert_array_equal(got, want)  # bit-exact integer logic


def test_make_attn_mask_noncausal_rows_ignore_q_index() -> None:
    num_masked = mx.array([1, 3])
    got = _np(make_attn_mask(4, num_masked, kv_length=6, causal=False))
    # Reference semantics: non-causal returns the un-broadcast (b,1,1,kv) form
    # (q index is simply not consulted).
    assert got.shape == (2, 1, 1, 6)
    # batch 0 attends kv 1..5; batch 1 attends kv 3..5
    assert np.all(got[0, 0, 0] == np.where(np.arange(6) >= 1, 0.0, -1e9))
    assert np.all(got[1, 0, 0] == np.where(np.arange(6) >= 3, 0.0, -1e9))


def test_make_segment_mask_direction() -> None:
    ids = mx.array([[0, 0, 1, 2], [1, 1, 1, 0]])
    got = _np(make_segment_mask(ids))
    assert got.shape == (2, 1, 4, 4)
    ids_np = np.asarray(ids)
    assert got[0, 0, 0, 1] == 0.0  # same segment → attend
    assert got[0, 0, 0, 2] == -1e9  # cross segment → blocked
    assert np.all(got == np.where(ids_np[:, None, :] == ids_np[:, :, None], 0.0, -1e9)[:, None])


# ---- MHA ----


def _tiny_cfg() -> TransformerConfig:
    return TransformerConfig(
        model_dims=32,
        hidden_dims=64,
        num_heads=4,
        attention_norm="rms",
        feedforward_norm="rms",
        qk_norm="rms",
        use_bias=False,
        use_rope_seq=True,
        use_rope_var=False,
        ff_activation="relu",
        deterministic=True,
    )


def test_mha_shapes_and_finiteness() -> None:
    m = MultiHeadAttention(num_heads=4, in_features=32, eps=1e-5)
    x = mx.array(np.random.default_rng(3).normal(size=(2, 9, 32)).astype(np.float32))
    patch = np.zeros((2, 9), dtype=bool)
    patch[0, 7:] = True  # trailing mask only: rows keep attendable KVs
    out, mask = m(x, patch_mask=mx.array(patch))
    mx.eval(out, mask)
    assert out.shape == (2, 9, 32)
    assert mask.shape == (2, 1, 9, 9)
    assert np.isfinite(np.asarray(out)).all()


def test_mha_all_linears_no_bias() -> None:
    m = MultiHeadAttention(num_heads=4, in_features=32, eps=1e-5)
    names = {k for k, _ in _flat(m)}
    assert names == {
        "query_proj.weight",
        "key_proj.weight",
        "value_proj.weight",
        "out_proj.weight",
        "query_ln.weight",
        "key_ln.weight",
        "per_dim_scale.per_dim_scale",
    }


def _flat(module):
    from mlx.utils import tree_flatten

    return tree_flatten(module.parameters())


# ---- MixingTransformer ----


def test_mixing_residual_form_zero_weights_is_identity() -> None:
    # post_ln(sublayer(0-output)) + x with ZERO sublayer weights ⇒ exactly + x
    # (RMSNorm of a zero vector is zero), for any input.
    cfg = _tiny_cfg()
    layer = MixingTransformer(cfg, eps=1e-5)
    flat = dict(_flat(layer))
    for k in list(flat):
        if "proj" in k or k.startswith(("ff0", "ff1")):
            flat[k] = mx.zeros_like(flat[k])
    from mlx.utils import tree_unflatten
    layer.update(tree_unflatten(list(flat.items())))
    x = mx.array(np.random.default_rng(4).normal(size=(2, 3, 5, 32)).astype(np.float32))
    patch = mx.zeros((2, 3, 5), dtype=mx.bool_)
    out, _ = layer(x, patch)
    np.testing.assert_allclose(_np(out), np.asarray(x), rtol=0, atol=2e-6)


def test_mixing_no_nans_with_random_weights() -> None:
    cfg = _tiny_cfg()
    layer = MixingTransformer(cfg, eps=1e-5)
    rng = np.random.default_rng(5)
    flat = {k: mx.array(rng.normal(0, 0.02, v.shape).astype(np.float32)) for k, v in _flat(layer)}
    from mlx.utils import tree_unflatten
    layer.update(tree_unflatten(list(flat.items())))
    x = mx.array(rng.normal(size=(1, 2, 6, 32)).astype(np.float32))
    patch = np.zeros((1, 2, 6), dtype=bool)
    patch[0, 0, 5:] = True
    out, _ = layer(x, mx.array(patch))
    assert np.isfinite(_np(out)).all()
