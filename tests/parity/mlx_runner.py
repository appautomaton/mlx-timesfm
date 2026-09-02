"""Parity runner for the MLX stack. Runs in the main .venv (pytest process or
`uv run python`). Forces CPU (SPEC A1) and stamps environment metadata into the
output npz, mirroring `torch_runner.py`.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import mlx.core as mx
import numpy as np


def relu(t: dict[str, mx.array]) -> dict[str, mx.array]:
    return {"y": mx.maximum(t["x"], 0)}


OPS: dict[str, Callable[[dict[str, mx.array]], dict[str, mx.array]]] = {
    "relu": relu,
}


def run_mlx(op: str, inputs_path: str, outputs_path: str) -> dict:
    mx.set_default_device(mx.cpu)

    data = np.load(inputs_path)
    tensors = {k: mx.array(data[k]) for k in data.files if k != "_meta"}
    out = OPS[op](tensors)
    mx.eval(*out.values())  # MLX is lazy — force computation before readback

    meta = {
        "stack": "mlx",
        "mlx_version": mx.__version__,
        "device": "cpu",
    }
    np.savez(
        outputs_path,
        **{k: np.ascontiguousarray(np.asarray(v)) for k, v in out.items()},
        _meta=np.array(json.dumps(meta)),
    )
    return {"op": op, "outputs": sorted(out), "meta": meta}
