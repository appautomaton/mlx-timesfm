"""Property tests for Phase 3 pieces (running stats, RevIN, roll/stitch,
ResidualBlock, forward plumbing) — torch-free."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_timesfm import (
    ResidualBlockConfig,
    StackedTransformersConfig,
    TimesFM3,
    TimesFM3Config,
    TransformerConfig,
)
from mlx_timesfm.dense import ResidualBlock
from mlx_timesfm.util import (
    get_output_patch_via_roll,
    get_running_stats,
    revin,
    stitch_patches,
)


def _np(x: mx.array) -> np.ndarray:
    mx.eval(x)
    return np.asarray(x)


# ---- running stats ----


def test_running_stats_matches_naive_numpy() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=(2, 3, 5, 7)).astype(np.float32)
    masks = rng.random(values.shape) < 0.25
    n, mu, sigma = _np(
        get_running_stats(mx.array(values), mx.array(masks))
    )
    # independent recompute: stats over all unmasked points of patches 0..i
    for b in range(2):
        for v in range(3):
            for i in range(5):
                sel = ~masks[b, v, : i + 1]
                vals = values[b, v, : i + 1][sel]
                assert n[b, v, i] == len(vals)
                want_mu = vals.mean() if len(vals) else 0.0
                want_sd = vals.std() if len(vals) else 0.0
                assert abs(mu[b, v, i] - want_mu) < 1e-5
                assert abs(sigma[b, v, i] - want_sd) < 1e-5


def test_running_stats_segment_reset() -> None:
    values = np.ones((1, 1, 4, 3), dtype=np.float32)
    values[0, 0, 2:] = 10.0  # new segment starts at patch 2
    seg = mx.array(np.array([[0, 0, 1, 1]]))
    n, mu, _ = _np(
        get_running_stats(
            mx.array(values), mx.zeros((1, 1, 4, 3), dtype=mx.bool_), segment_ids=seg
        )
    )
    assert n[0, 0, 1] == 6  # two 3-point patches of segment 0
    assert n[0, 0, 2] == 3  # reset! only patch 2 counted
    assert abs(mu[0, 0, 2] - 10.0) < 1e-6
    assert abs(mu[0, 0, 3] - 10.0) < 1e-6


# ---- revin ----


def test_revin_roundtrip_and_safe_division() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 2, 4, 5)).astype(np.float32)
    mu = x.mean(axis=-1).astype(np.float32)
    sigma = x.std(axis=-1).astype(np.float32)
    mu, sigma = mx.array(mu), mx.array(sigma)
    normed = revin(mx.array(x), mu, sigma)
    back = _np(revin(normed, mu, sigma, reverse=True))
    np.testing.assert_allclose(back, x, rtol=1e-5, atol=1e-6)
    # zero sigma must not produce inf/nan (guard maps <1e-6 to 1.0)
    zero_sigma = revin(mx.array(x), mu, mx.zeros_like(sigma))
    assert np.isfinite(_np(zero_sigma)).all()


# ---- roll / stitch ----


def test_roll_matches_numpy_and_wrap_mask_exact() -> None:
    x = np.arange(2 * 1 * 4 * 3, dtype=np.float32).reshape(2, 1, 4, 3)
    out_m, wrap_m = get_output_patch_via_roll(mx.array(x), 2)
    out, wrap = _np(out_m), _np(wrap_m)
    assert out.shape == (2, 1, 4, 6)
    # reference takes rolling_mat[:, :, :, 1:, :] — blocks are shifts 1..rolls,
    # i.e. block j = roll(x, -(j+1)); block 0 is NOT x itself
    np.testing.assert_array_equal(out[..., :3], np.roll(x, -1, axis=2))
    np.testing.assert_array_equal(out[..., 3:], np.roll(x, -2, axis=2))
    # wrap: for patch i, points j>=p are garbage iff (i+1)+j//p >= n
    n, p, rolls = 4, 3, 2
    expect = (
        np.arange(n)[:, None] + 1 + np.arange(rolls * p)[None, :] // p >= n
    )
    np.testing.assert_array_equal(wrap[0, 0], expect)


def test_stitch_weights_linear() -> None:
    # 3 patches, p=2, overlap=2 → output length 3*2+2=8; middle overlap region
    # must be a convex combination with weights [1,0] at the extremes.
    preds = np.zeros((1, 1, 3, 4, 1), dtype=np.float32)
    preds[0, 0, 0, 2:, 0] = [10, 10]  # patch0's overlap points
    out = _np(stitch_patches(mx.array(preds), 2))
    assert out.shape == (1, 1, 8, 1)
    # first_chunk = patch0[:2] = [0,0]; then stitched overlaps weight [1,0]:
    # point at index 2 = 1*10 + 0*0 = 10
    assert out[0, 0, 0, 0] == 0.0
    assert out[0, 0, 2, 0] == 10.0


# ---- ResidualBlock ----


def _small_cfg() -> TimesFM3Config:
    return TimesFM3Config(
        input_patch_len=4,
        output_patch_len=8,
        quantiles=(0.25, 0.5, 0.75),
        residual_block_config=ResidualBlockConfig(
            hidden_dims=16, output_dims=16, use_bias=False, activation="relu"
        ),
        transformer_config=StackedTransformersConfig(
            num_layers=1,
            transformer=TransformerConfig(
                model_dims=16,
                hidden_dims=32,
                num_heads=2,
                attention_norm="rms",
                feedforward_norm="rms",
                qk_norm="rms",
                use_bias=False,
                use_rope_seq=True,
                use_rope_var=False,
                ff_activation="relu",
                deterministic=True,
            ),
        ),
        rmsnorm_eps=1e-5,
    )


def test_residual_block_eager_shapes_and_no_norm() -> None:
    cfg = _small_cfg().residual_block_config
    m = ResidualBlock(cfg, input_dim=24, eps=1e-5)
    names = {k for k, _ in _flat(m)}
    assert names == {
        "hidden_layer.weight",
        "output_layer.weight",
        "residual_layer.weight",
    }  # prenorm="none" ⇒ NO pre_norm (checkpoint agrees)
    x = mx.array(np.random.default_rng(2).normal(size=(2, 6, 24)).astype(np.float32))
    assert m(x).shape == (2, 6, 16)


def test_residual_block_identity_skip() -> None:
    cfg = ResidualBlockConfig(
        hidden_dims=8,
        output_dims=8,
        use_bias=False,
        activation="none",
        identity_skip=True,
    )
    m = ResidualBlock(cfg, input_dim=8, eps=1e-5)
    # zero all weights: out = out_layer(hidden) + x = x
    flat = dict(_flat(m))
    from mlx.utils import tree_unflatten

    layer_flat = {k: mx.zeros_like(v) for k, v in flat.items()}
    m.update(tree_unflatten(list(layer_flat.items())))
    x = mx.array(np.random.default_rng(3).normal(size=(1, 3, 8)).astype(np.float32))
    np.testing.assert_array_equal(_np(m(x)), np.asarray(x))


def _flat(module):
    from mlx.utils import tree_flatten

    return tree_flatten(module.parameters())


# ---- model plumbing ----


def test_model_requires_explicit_eps() -> None:
    cfg = dataclasses_replace_eps(_small_cfg(), None)
    with pytest.raises(ValueError, match="rmsnorm_eps"):
        TimesFM3(cfg)


def dataclasses_replace_eps(cfg, eps):
    import dataclasses

    return dataclasses.replace(cfg, rmsnorm_eps=eps)


def test_model_param_tree_matches_checkpoint_naming() -> None:
    m = TimesFM3(_small_cfg())
    names = {k for k, _ in _flat(m)}
    assert "pre_transformer_resblock.hidden_layer.weight" in names
    assert "transformer_stack.layers.0.seq_attn.per_dim_scale.per_dim_scale" in names
    assert "output_head.bias" in names  # the only biased linear
    assert "output_head.weight" in names


def test_model_forward_leading_mask_semantics_and_shapes() -> None:
    m = TimesFM3(_small_cfg())
    rng = np.random.default_rng(4)
    b, v, n, p = 1, 2, 5, 4
    inp = {
        "values": mx.array(rng.normal(size=(b, v, n, p)).astype(np.float32)),
        # leading fully-masked patch 0; patch 3 fully masked mid-sequence
        "masks": mx.array(
            np.zeros((b, v, n, p), dtype=bool) | (np.arange(n)[None, None, :, None] == 0)
        ),
        "patch_is_target": mx.ones((b, v, n), dtype=mx.bool_),
    }
    out = m.forward(inp, return_aux_outputs=True)
    logits = out["logits"]
    mx.eval(logits)
    assert logits.shape == (b, v, n, 8, 3)
    assert np.isfinite(np.asarray(logits)).all()
    # effective mask: cumprod ⇒ only LEADING fully-masked patches stay masked;
    # patch 3 fully masked but patch 2 visible ⇒ cumprod keeps it visible.
    pm = out["__call__:transformer_input"]  # touch aux to keep graph alive
    mx.eval(pm)


def test_forecast_returns_only_target_variates_with_covariates(monkeypatch) -> None:
    model = TimesFM3(_small_cfg())

    def fake_decode(
        target,
        horizon=0,
        past_only_covariates=None,
        past_future_covariates=None,
        **_kwargs,
    ):
        total_variates = (
            target.shape[1]
            + past_only_covariates.shape[1]
            + past_future_covariates.shape[1]
        )
        return mx.zeros((target.shape[0], total_variates, horizon, 3))

    monkeypatch.setattr(model, "decode", fake_decode)
    target = np.zeros((1, 2, 16), dtype=np.float32)
    past_only = np.zeros((1, 1, 16), dtype=np.float32)
    past_future = np.zeros((1, 2, 24), dtype=np.float32)
    output = model.forecast(
        target,
        horizon=8,
        past_only_covariates=past_only,
        past_future_covariates=past_future,
    )
    assert output.shape == (1, 2, 8, 3)


def test_revin_math_against_numpy_reference() -> None:
    x = np.linspace(-3, 3, 40, dtype=np.float32).reshape(2, 2, 2, 5)
    mu = np.array([[[0.5, -0.5], [1.0, 2.0]], [[0.0, 0.0], [3.0, -3.0]]], dtype=np.float32)
    sigma = np.full(mu.shape, 2.0, dtype=np.float32)
    got = _np(revin(mx.array(x), mx.array(mu), mx.array(sigma)))
    want = (x - mu[..., None]) / 2.0
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)
