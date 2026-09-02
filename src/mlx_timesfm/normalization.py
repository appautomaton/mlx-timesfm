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

"""Normalization layers for the MLX port.

Mirrors ``timesfm3.normalization`` (+ the RMSNorm the torch reference gets from
``torch.nn``), with one deliberate difference: eps is a REQUIRED explicit
constructor argument (SPEC R5 — framework defaults differ by version and we
refuse to silently depend on either one).
"""

from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn

# Same constant as the reference (normalization.py); not a "recalled" value —
# it is defined by the reference source we port.
_RECIPROCAL_OF_SOFTPLUS_0 = 1.442695041

__all__ = ["RMSNorm", "PerDimScale"]


class RMSNorm(nn.Module):
    """RMSNorm matching ``torch.nn.RMSNorm`` semantics: x * rsqrt(mean(x²)+eps) * weight.

    ``eps`` must be passed explicitly (SPEC R5); ``None`` raises.
    """

    def __init__(self, dims: int, eps: float | None = None, affine: bool = True):
        super().__init__()
        if eps is None:
            raise ValueError(
                "RMSNorm eps must be given explicitly (SPEC R5): never rely on "
                "framework defaults. Use mlx_timesfm.config.INFERENCE_RMSNORM_EPS "
                "for inference; parity passes the probed reference value."
            )
        self.dims = dims
        self.eps = eps
        if affine:
            self.weight = mx.ones((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        scale = mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)
        out = x * scale
        if hasattr(self, "weight"):
            out = out * self.weight
        return out


class PerDimScale(nn.Module):
    """Per-dimension scaling (Pax-style), port of the reference PerDimScale.

    ``x * RECIPROCAL_OF_SOFTPLUS_0 / sqrt(d) * softplus(per_dim_scale)``

    Parameter name is ``per_dim_scale`` (checkpoint key:
    ``…per_dim_scale.per_dim_scale``). Reference init is zeros, making the net
    scale ≈ 1/√d at init; the checkpoint carries learned values.
    """

    def __init__(self, num_dims: int):
        super().__init__()
        self.num_dims = num_dims
        self.per_dim_scale = mx.zeros((num_dims,))

    def __call__(self, x: mx.array) -> mx.array:
        return (
            x
            * _RECIPROCAL_OF_SOFTPLUS_0
            / math.sqrt(self.num_dims)
            * _softplus(self.per_dim_scale)
        )


def _softplus(x: mx.array) -> mx.array:
    """softplus mirroring torch.nn.functional.softplus defaults (beta=1, threshold=20).

    (mx.softplus does not exist in mlx 0.32.)
    """
    return mx.where(x > 20.0, x, mx.log1p(mx.exp(x)))
