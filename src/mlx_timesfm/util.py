"""Small shared utilities for the MLX port."""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten


def get_activation(name: str) -> Callable[[mx.array], mx.array]:
    """Mirror of ``timesfm3.util.get_activation_fn`` for the options we support."""
    if name == "relu":
        return lambda x: mx.maximum(x, 0)  # mx.relu doesn't exist in mlx 0.32
    if name == "none":
        return lambda x: x
    raise NotImplementedError(
        f"activation {name!r} not ported (reference supports relu/swish/none)"
    )


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
