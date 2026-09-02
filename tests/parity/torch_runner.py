#!/usr/bin/env python
"""Parity runner for the PyTorch reference stack.

Runs ONLY under `.venv-torch/bin/python` (torch lives there; the main .venv and
the package itself must never see torch). Invoked as a subprocess by
`bridge.py`:

    .venv-torch/bin/python tests/parity/torch_runner.py <op> <inputs.npz> <outputs.npz>

Forces CPU (SPEC A1: same-device comparison — a diff must implicate code, not
kernels) and stamps environment metadata into the output npz.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch


def relu(t: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    x = torch.from_numpy(t["x"])
    return {"y": torch.nn.functional.relu(x).numpy()}


OPS: dict[str, Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]] = {
    "relu": relu,
}


def main(op: str, inputs_path: str, outputs_path: str) -> None:
    torch.set_grad_enabled(False)
    # A1: CPU only. Not os.environ-based so we fail loudly if it didn't take.
    torch.set_num_threads(max(1, torch.get_num_threads()))

    tensors = dict(np.load(inputs_path))
    out = OPS[op]({k: v for k, v in tensors.items()})

    meta = {
        "stack": "torch",
        "torch_version": torch.__version__,
        # SPEC R5 probe: record whatever the reference env's default actually
        # is (torch 2.13 reports None) — never cite a remembered value.
        "torch_rmsnorm_eps_default": torch.nn.RMSNorm(80).eps,
        "device": "cpu",
    }
    np.savez(
        outputs_path,
        **{k: np.ascontiguousarray(np.asarray(v)) for k, v in out.items()},
        _meta=np.array(json.dumps(meta)),
    )
    print(json.dumps({"op": op, "outputs": sorted(out), "meta": meta}))


if __name__ == "__main__":
    assert len(sys.argv) == 4, "usage: torch_runner.py <op> <inputs.npz> <outputs.npz>"
    assert Path(sys.argv[2]).exists(), sys.argv[2]
    main(*sys.argv[1:])
