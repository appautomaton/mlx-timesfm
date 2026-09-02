"""torch ↔ mlx parity bridge.

One seeded generator writes **inputs and (future) model weights** to a .npz so
both stacks start bit-identical; the torch side runs as a subprocess under
`.venv-torch/bin/python`, the mlx side runs in-process; a comparator diffs the
outputs into `.agents/parity-reports/<op>.md`.

Both runners force CPU (SPEC A1). Every report header carries the setup probe:
torch/mlx versions and the reference env's actual RMSNorm eps default (SPEC R5).

Artifacts (.npz) go to `tests/parity/artifacts/` (gitignored).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "tests/parity/artifacts"
REPORTS = ROOT / ".agents/parity-reports"
TORCH_PY = ROOT / ".venv-torch/bin/python"
_HERE = Path(__file__).resolve().parent


def torch_env_available() -> bool:
    return TORCH_PY.exists()


# --------------------------------------------------------------------------
# Seeded case generation (numpy — no framework on either side, so both stacks
# read bit-identical tensors out of the npz)
# --------------------------------------------------------------------------

CaseSpec = dict[str, Callable[[np.random.Generator], np.ndarray]]

CASES: dict[str, CaseSpec] = {
    # Smoke op: mix of signs, an exact zero and exact negatives/positives so a
    # relu difference cannot hide in random magnitudes.
    "relu": {
        "x": lambda rng: np.concatenate(
            [rng.normal(size=4095), np.zeros(1)]
        ).astype(np.float32)
    },
}


def gen_case(op: str, seed: int, out_dir: Path = ARTIFACTS) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arrays = {name: fn(rng).astype(np.float32) for name, fn in CASES[op].items()}
    path = out_dir / f"{op}_seed{seed}_inputs.npz"
    np.savez(path, **arrays)
    return path


# --------------------------------------------------------------------------
# Runners
# --------------------------------------------------------------------------


def run_torch(op: str, inputs: Path, outputs: Path) -> dict[str, Any]:
    """Torch side, in its own interpreter (main venv must stay torch-free)."""
    r = subprocess.run(
        [str(TORCH_PY), str(_HERE / "torch_runner.py"), op, str(inputs), str(outputs)],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"torch_runner failed:\n{r.stdout}\n{r.stderr}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def run_mlx(op: str, inputs: Path, outputs: Path) -> dict[str, Any]:
    if str(_HERE) not in sys.path:  # work when invoked from any cwd
        sys.path.insert(0, str(_HERE))
    from mlx_runner import run_mlx as _run

    return _run(op, str(inputs), str(outputs))


# --------------------------------------------------------------------------
# Compare + report
# --------------------------------------------------------------------------


def compare(torch_npz: Path, mlx_npz: Path) -> dict[str, dict[str, Any]]:
    t, m = np.load(torch_npz), np.load(mlx_npz)
    keys = [k for k in t.files if k != "_meta"]
    if sorted(keys) != sorted(k for k in m.files if k != "_meta"):
        raise AssertionError(f"output key mismatch: torch={keys} mlx={m.files}")
    per: dict[str, dict[str, Any]] = {}
    for k in keys:
        a, b = t[k], m[k]
        if a.shape != b.shape:
            raise AssertionError(f"{k}: shape {a.shape} vs {b.shape}")
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        per[k] = {
            "shape": list(a.shape),
            "max_abs_diff": float(diff.max()) if a.size else 0.0,
            "bit_exact": bool(np.array_equal(a, b)),
        }
    return per


def write_report(
    op: str,
    seed: int,
    per: dict[str, dict[str, Any]],
    metas: list[dict[str, Any]],
    tol: float | None,
    require_bit_exact: bool,
) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"{op}.md"
    ok = all(
        (v["bit_exact"] if require_bit_exact else (tol is not None and v["max_abs_diff"] <= tol))
        for v in per.values()
    )
    lines = [
        f"# Parity report: `{op}`",
        "",
        f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- seed: {seed}",
        "- devices: torch=cpu, mlx=cpu (SPEC A1)",
        f"- criterion: {'bit-exact' if require_bit_exact else f'max abs diff <= {tol}'}",
        f"- **verdict: {'PASS' if ok else 'FAIL'}**",
        "",
        "## Environment probe (SPEC R5)",
        "",
        "| stack | version | rmsnorm eps default | device |",
        "|---|---|---|---|",
    ]
    for m in metas:
        ver = m.get("torch_version") or m.get("mlx_version")
        eps = m.get("torch_rmsnorm_eps_default", "n/a")
        lines.append(f"| {m['stack']} | {ver} | {eps} | {m['device']} |")
    lines += [
        "",
        "## Outputs",
        "",
        "| output | shape | bit-exact | max abs diff |",
        "|---|---|---|---|",
    ]
    for k in sorted(per):
        v = per[k]
        lines.append(f"| `{k}` | {tuple(v['shape'])} | {v['bit_exact']} | {v['max_abs_diff']:g} |")
    lines.append("")
    path.write_text("\n".join(lines))
    if not ok:
        raise AssertionError(f"parity FAILED for {op}; report at {path}")
    return path


def run_parity_case(
    op: str,
    seed: int = 0,
    tol: float | None = None,
    require_bit_exact: bool = False,
) -> dict[str, Any]:
    """Full pipeline for one op; raises AssertionError on parity failure."""
    assert tol is not None or require_bit_exact, "pass tol= or require_bit_exact=True"
    inputs = gen_case(op, seed)
    t_out = ARTIFACTS / f"{op}_seed{seed}_torch.npz"
    m_out = ARTIFACTS / f"{op}_seed{seed}_mlx.npz"
    t_meta = run_torch(op, inputs, t_out)["meta"]
    m_meta = run_mlx(op, inputs, m_out)["meta"]
    per = compare(t_out, m_out)
    report = write_report(op, seed, per, [t_meta, m_meta], tol, require_bit_exact)
    return {"per": per, "report": report}
