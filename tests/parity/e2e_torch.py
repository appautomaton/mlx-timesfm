#!/usr/bin/env python
"""A2 end-to-end REFERENCE runner (torch side). Runs ONLY under
`.venv-torch/bin/python`, invoked as a subprocess by test_parity_e2e_a2.py.

Decodes the five committed A2 fixtures (SPEC §5) at the reference torch stack
and dumps (1, v, h, q) logits + per-channel context sigma per cell as npz.

Reuses `torch_runner._build_real_model` — the SAME deliberate deviations as
the real-weights forward parity run: `use_sdpa=False` (manual branch, so the
port's manual attention is compared against identical math) and RMSNorm eps
force-set to PARITY_EPS on every module (R5). This run therefore does NOT
claim parity against the untouched official configuration (SDPA-on, torch's
own eps); it is labelled exactly that way in the A2 report.

Usage: .venv-torch/bin/python e2e_torch.py <out_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from torch_runner import CKPT_DIR, PARITY_EPS, _build_real_model

torch.set_grad_enabled(False)

HERE = Path(__file__).resolve().parent
FIXTURES_DIR = HERE.parent / "fixtures"
HORIZONS = (32, 128, 512)
CTX_LEN = {"ar1_mv3": 1024}  # SPEC A2: multivariate set uses context 1024


def run(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = _build_real_model()

    for csv in sorted(FIXTURES_DIR.glob("*.csv")):
        stem = csv.stem
        data = np.loadtxt(csv, delimiter=",", skiprows=1, ndmin=2)  # (T, v)
        context = CTX_LEN.get(stem, 512)
        assert context + max(HORIZONS) <= data.shape[0], stem
        # Context = the FIRST `context` points (the fixture step/trend must be
        # inside it — e.g. near_flat's step at t=100); the tail of the series
        # is what horizons up to 512 forecast into.
        ctx = data[:context].T.astype(np.float32)[None]  # (1, v, context)
        # sigma := population std of unmasked context values, per target series
        sigma = ctx[0].std(axis=-1, dtype=np.float64)
        target = torch.from_numpy(np.ascontiguousarray(ctx))
        for h in HORIZONS:
            logits = model.decode(target, horizon=h)
            np.savez(
                out_dir / f"{stem}_h{h}.npz",
                logits=logits.numpy(),
                sigma=sigma,
                context=np.int64(context),
            )

    meta = {
        "stack": "torch",
        "torch_version": torch.__version__,
        "torch_rmsnorm_eps_default": torch.nn.RMSNorm(80).eps,
        "effective_rmsnorm_eps": PARITY_EPS,
        "attention_path": "reference manual (use_sdpa=False, rescale_logits=False)",
        "weights": str(CKPT_DIR),
        "device": "cpu",
        "fixtures": [c.stem for c in sorted(FIXTURES_DIR.glob("*.csv"))],
        "horizons": list(HORIZONS),
    }
    (out_dir / "meta_torch.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    assert len(sys.argv) == 2, "usage: e2e_torch.py <out_dir>"
    run(Path(sys.argv[1]))
