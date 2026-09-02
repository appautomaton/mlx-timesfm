"""Dense layers for the MLX port.

Port of ``timesfm3.dense.ResidualBlock`` with one deliberate difference:
the reference rebuilds its linears lazily on first forward
(``set_input_dims``); we take ``input_dim`` eagerly at construction
(the model knows it: ``2 * (input_patch_len + output_patch_len)``, i.e. 192
for the official checkpoint). Parameter names/shapes are identical to the
reference state dict: ``hidden_layer`` / ``output_layer`` /
``residual_layer`` (present iff ``identity_skip=False``) / ``pre_norm``
(present iff ``prenorm="rms"`` — the official checkpoint has none).
"""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

from .config import ResidualBlockConfig
from .normalization import RMSNorm
from .util import get_activation


class ResidualBlock(nn.Module):
    """Two-layer MLP + linear (or identity) skip.

    ``hidden = act(hidden_layer(x_norm)); out = output_layer(hidden) + skip(x)``
    where ``x_norm`` is RMSNorm(x) iff prenorm=="rms" — GOTCHA: with the
    official config (prenorm="none") there is NO norm here.
    """

    def __init__(self, config: ResidualBlockConfig, *, input_dim: int, eps: float):
        super().__init__()
        self.config = config
        self.hidden_layer = nn.Linear(input_dim, config.hidden_dims, bias=config.use_bias)
        self.output_layer = nn.Linear(
            config.hidden_dims, config.output_dims, bias=config.use_bias
        )
        self.residual_layer = (
            None
            if config.identity_skip
            else nn.Linear(input_dim, config.output_dims, bias=config.use_bias)
        )
        self.pre_norm = RMSNorm(input_dim, eps) if config.prenorm == "rms" else None
        self.activation = get_activation(config.activation)

    def __call__(self, x: mx.array) -> mx.array:
        hidden_input = self.pre_norm(x) if self.pre_norm is not None else x
        hidden = self.activation(self.hidden_layer(hidden_input))
        skip = x if self.residual_layer is None else self.residual_layer(x)
        return self.output_layer(hidden) + skip
