"""Checkpoint loading (SPEC F3/F5).

``load()`` reads ``config.json`` + ``model.safetensors`` straight from the
original checkpoint directory — MLX reads safetensors natively, so there is no
converted intermediate artifact. Parameter names map 1:1 onto the module tree
(verified against the 445-tensor state dict); loading is STRICT (R6): any
missing/extra key or shape mismatch raises, never a silent partial load.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from .config import load_config
from .model import TimesFM3
from .util import load_parameters

__all__ = ["load"]

WEIGHTS_FILENAME = "model.safetensors"


def load(
    path: str | Path = "models/timesfm_3_0/original",
    *,
    rmsnorm_eps: float | None = None,
) -> TimesFM3:
    """Load a TimesFM 3.0 checkpoint directory into an inference-ready model.

    Args:
      path: directory holding ``config.json`` + ``model.safetensors``.
      rmsnorm_eps: overrides the documented ``INFERENCE_RMSNORM_EPS`` port
        constant applied at this inference boundary (SPEC R5); pass the parity
        harness's effective value to reproduce reference-env numerics.
    """
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} is not a checkpoint directory")
    weights = directory / WEIGHTS_FILENAME
    if not weights.is_file():
        raise FileNotFoundError(f"{weights} not found")

    config = load_config(directory, rmsnorm_eps=rmsnorm_eps)
    model = TimesFM3(config)
    load_parameters(model, mx.load(str(weights)))
    return model
