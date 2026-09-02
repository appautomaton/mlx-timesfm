#!/usr/bin/env python3
"""Smoke-test an installed distribution without local source or weights."""

from __future__ import annotations

import os
from importlib.metadata import version

import mlx_timesfm


def main() -> None:
    assert mlx_timesfm.__version__ == version("mlx-timesfm")
    assert mlx_timesfm.__version__ == "0.0.1"
    assert os.environ["MLX_ENABLE_TF32"] == "0"
    assert callable(mlx_timesfm.load)
    assert mlx_timesfm.TimesFM3Config().num_quantiles == 9
    print(f"mlx-timesfm {mlx_timesfm.__version__} distribution smoke test passed")


if __name__ == "__main__":
    main()
