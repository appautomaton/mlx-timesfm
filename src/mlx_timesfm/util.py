"""Small shared utilities for the MLX port.

Ports ``timesfm3.util`` (read-only PyTorch reference): activation lookup,
masked running statistics, RevIN, output-patch rolling, patch stitching —
plus our strict state-dict loader. ``DecodeCache`` is deliberately absent
(Phase 4); ``load_safetensors`` is replaced by ``mx.load``.
"""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

_TOLERANCE = 1e-6  # mirrors reference util._TOLERANCE


def get_activation(name: str) -> Callable[[mx.array], mx.array]:
    """Mirror of ``timesfm3.util.get_activation_fn`` for the options we support."""
    if name == "relu":
        return lambda x: mx.maximum(x, 0)  # mx.relu doesn't exist in mlx 0.32
    if name == "none":
        return lambda x: x
    raise NotImplementedError(
        f"activation {name!r} not ported (reference supports relu/swish/none)"
    )


# --------------------------------------------------------------------------
# Running statistics & RevIN (ports of timesfm3.util)
# --------------------------------------------------------------------------


def _make_safe_for_division(values: mx.array) -> mx.array:
    """Mirror of reference: near-zero std must not blow up the division."""
    return mx.where(values < _TOLERANCE, 1.0, values)


def update_running_stats(
    n: mx.array,
    mu: mx.array,
    sigma: mx.array,
    x: mx.array,
    mask: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """One patch of masked-parallel-update stats. (b,v) stats, x/mask (b,v,p).

    Mirrors ``util.update_running_stats`` expression-for-expression
    (masked parallel-variance combination, not a naive recompute).
    """
    is_legit = ~mask
    inc_n = is_legit.astype(mx.float32).sum(axis=-1)
    x_masked = mx.where(is_legit, x, mx.zeros_like(x))
    inc_sum = x_masked.sum(axis=-1)
    inc_mu = mx.where(inc_n == 0, mx.zeros_like(inc_sum), inc_sum / inc_n)
    x_diff_sq = mx.where(
        is_legit, (x - inc_mu[..., None]) ** 2, mx.zeros_like(x)
    )
    inc_var = mx.where(
        inc_n == 0, mx.zeros_like(inc_sum), x_diff_sq.sum(axis=-1) / inc_n
    )
    inc_sigma = mx.sqrt(inc_var)

    new_n = n + inc_n
    new_mu = mx.where(new_n == 0, mx.zeros_like(mu), (n * mu + inc_mu * inc_n) / new_n)
    new_sigma = mx.sqrt(
        mx.where(
            new_n == 0,
            mx.zeros_like(sigma),
            (
                n * sigma * sigma
                + inc_n * inc_sigma * inc_sigma
                + n * (mu - new_mu) * (mu - new_mu)
                + inc_n * (inc_mu - new_mu) * (inc_mu - new_mu)
            )
            / new_n,
        )
    )
    return new_n, new_mu, new_sigma


def get_running_stats(
    values: mx.array,
    masks: mx.array,
    *,
    segment_ids: mx.array | None = None,
    initial_stats: tuple[mx.array, mx.array, mx.array] | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Cumulative per-patch stats; each patch i sees patches 0..i inclusive.

    Python loop over patches (reference does the same); cumsum closed form is
    a later optimisation. Returns three (b, v, n) arrays.
    """
    b, v, n, _ = values.shape
    zeros = mx.zeros((b, v), dtype=mx.float32)
    if initial_stats is None:
        init_n = init_mu = init_sigma = zeros
    else:
        init_n, init_mu, init_sigma = initial_stats

    if segment_ids is None:
        is_new_segment = mx.zeros((b, n), dtype=mx.bool_)
    else:
        # F.pad(x[:, :-1], (1, 0), value=-1): prepend a -1 column.
        neg_one = mx.full((b, 1), -1, dtype=segment_ids.dtype)
        shifted = mx.concatenate([neg_one, segment_ids[:, :-1]], axis=1)
        is_new_segment = segment_ids != shifted

    all_n, all_mu, all_sigma = [], [], []
    cur_n, cur_mu, cur_sigma = init_n, init_mu, init_sigma
    for i in range(n):
        reset = is_new_segment[:, i][:, None]  # (b, 1) broadcasts over v
        cur_n = mx.where(reset, init_n, cur_n)
        cur_mu = mx.where(reset, init_mu, cur_mu)
        cur_sigma = mx.where(reset, init_sigma, cur_sigma)
        cur_n, cur_mu, cur_sigma = update_running_stats(
            cur_n, cur_mu, cur_sigma, values[:, :, i, :], masks[:, :, i, :]
        )
        all_n.append(cur_n)
        all_mu.append(cur_mu)
        all_sigma.append(cur_sigma)
    return (
        mx.stack(all_n, axis=2),
        mx.stack(all_mu, axis=2),
        mx.stack(all_sigma, axis=2),
    )


def revin(
    x: mx.array, mu: mx.array, sigma: mx.array, reverse: bool = False
) -> mx.array:
    """Reversible per-instance norm; mu/sigma have 1 or 2 fewer dims than x."""
    if mu.ndim == x.ndim - 1:
        mu = mu[..., None]
        sigma = sigma[..., None]
    elif mu.ndim == x.ndim - 2:
        mu = mu[..., None, None]
        sigma = sigma[..., None, None]
    else:
        raise ValueError(f"Unsupported shapes for x and mu: {x.shape}, {mu.shape}.")
    if reverse:
        return x * sigma + mu
    return (x - mu) / _make_safe_for_division(sigma)


def get_output_patch_via_roll(
    x: mx.array, rolls: int
) -> tuple[mx.array, mx.array]:
    """(b, v, n, p) → rolled output patches (b, v, n, p*rolls) + wrap mask.

    Chained ``torch.roll(shifts=-1, dims=2)`` == one roll by -(i+1); the
    reference fills a rolling_mat the same way. Wrap mask True = the source
    timepoint fell off the end of the sequence (wrap-around garbage).
    """
    _, _, n, p = x.shape
    shifts = [mx.roll(x, -(i + 1), axis=2) for i in range(rolls)]
    result = mx.concatenate(shifts, axis=3)

    patch_idx = mx.arange(n)[:, None]
    point_idx = mx.arange(rolls * p)[None, :]
    source_patch = patch_idx + 1 + point_idx // p
    wrap_mask = (source_patch >= n)[None, None, :, :]
    return result, wrap_mask


def stitch_patches(patch_preds: mx.array, patch_len: int) -> mx.array:
    """Linear stitching of overlapping patch predictions.

    (b, v, num_patches, patch_len + overlap, q) →
    (b, v, num_patches * patch_len + overlap, q). Exact port of the reference
    slicing math.
    """
    b, v, num_patches, total_len, q = patch_preds.shape
    overlap = total_len - patch_len
    if num_patches == 1:
        return patch_preds[:, :, 0, :, :]

    stitch_weights = mx.linspace(1.0, 0.0, overlap).reshape(1, 1, 1, overlap, 1)

    first_chunk = patch_preds[:, :, 0, :patch_len, :]
    prev_patches = patch_preds[:, :, :-1, :, :]
    next_patches = patch_preds[:, :, 1:, :, :]

    prev_overlaps = prev_patches[:, :, :, patch_len:, :]
    next_overlaps = next_patches[:, :, :, :overlap, :]
    stitched_overlaps = stitch_weights * prev_overlaps + (
        1.0 - stitch_weights
    ) * next_overlaps
    middles = next_patches[:, :, :, overlap:patch_len, :]

    output_chunks = mx.concatenate([stitched_overlaps, middles], axis=3)
    mid = output_chunks.reshape(b, v, (num_patches - 1) * patch_len, q)
    tail = patch_preds[:, :, -1, patch_len:, :]
    return mx.concatenate([first_chunk, mid, tail], axis=2)


def load_parameters(model: nn.Module, params: dict[str, mx.array]) -> None:
    """Strict state-dict loader (SPEC R6).

    Unlike ``nn.Module.update``, this RAISES on any missing key, unexpected
    extra key, or shape mismatch — never a silent partial load.
    """
    have = dict(tree_flatten(model.parameters()))
    missing = sorted(set(have) - set(params))
    extra = sorted(set(params) - set(have))
    bad_shape = sorted(
        k for k in set(have) & set(params) if have[k].shape != params[k].shape
    )
    if missing or extra or bad_shape:
        raise ValueError(
            f"state dict mismatch: missing={missing[:8]} extra={extra[:8]} "
            f"shape_mismatch={bad_shape[:8]} "
            f"(n_missing={len(missing)} n_extra={len(extra)} n_bad_shape={len(bad_shape)})"
        )
    model.update(tree_unflatten(list(params.items())))
