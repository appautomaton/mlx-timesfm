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

"""MLX port of Google TimesFM 3.0 (inference only)."""

import os as _os
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version

# FP32 is the supported parity baseline. MLX may otherwise select reduced-
# precision matrix kernels while retaining float32 input/output dtypes. This
# must be set before the first MLX compute; an explicit user setting still wins.
_os.environ.setdefault("MLX_ENABLE_TF32", "0")

try:
    __version__ = _distribution_version("mlx-timesfm")
except _PackageNotFoundError:  # source checkout used without installation
    __version__ = "0.0.0+unknown"

from .config import (
    INFERENCE_RMSNORM_EPS,
    ResidualBlockConfig,
    StackedTransformersConfig,
    TimesFM3Config,
    TransformerConfig,
    load_config,
)
from .loader import load
from .model import TimesFM3

__all__ = [
    "__version__",
    "INFERENCE_RMSNORM_EPS",
    "ResidualBlockConfig",
    "StackedTransformersConfig",
    "TimesFM3",
    "TimesFM3Config",
    "TransformerConfig",
    "load",
    "load_config",
]
