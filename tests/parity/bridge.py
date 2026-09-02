"""torch ↔ mlx parity bridge.

One seeded numpy generator writes **inputs AND model weights** to .npz files so
both stacks start bit-identical; the torch side runs as a subprocess under
`.venv-torch/bin/python` (the pytest process itself must never import torch),
the mlx side runs in-process; a comparator diffs outputs into
`.agents/parity-reports/<case>.md`.

Both runners force CPU (SPEC A1). Every report header carries the setup probe:
torch/mlx versions, the reference env's RMSNorm eps default, and the EFFECTIVE
eps actually used on both sides (SPEC R5).

Case definitions (shapes, seeds' generators, A1 gates) live in `cases.py`
(pure numpy, importable by both environments). Artifacts (.npz) go to
`tests/parity/artifacts/` (gitignored).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from cases import CASES

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "tests/parity/artifacts"
REPORTS = ROOT / ".agents/parity-reports"
TORCH_PY = ROOT / ".venv-torch/bin/python"
_HERE = Path(__file__).resolve().parent


def torch_env_available() -> bool:
    return TORCH_PY.exists()


def gen_case(case: str, seed: int, out_dir: Path = ARTIFACTS) -> tuple[Path, str]:
    """Write <case>_seed<s>_inputs.npz (+ _weights.npz if the case has params).

    Returns (inputs_path, weights_path_or_dash).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = CASES[case]
    rng = np.random.default_rng(seed)
    inputs_path = out_dir / f"{case}_seed{seed}_inputs.npz"
    np.savez(inputs_path, **spec.inputs(rng))
    if spec.weights is None:
        return inputs_path, "-"
    weights_path = out_dir / f"{case}_seed{seed}_weights.npz"
    np.savez(weights_path, **spec.weights(np.random.default_rng(seed + 10_000)))
    return inputs_path, str(weights_path)


# --------------------------------------------------------------------------
# Runners
# --------------------------------------------------------------------------


def run_torch(case: str, inputs: Path, weights: str, outputs: Path) -> dict[str, Any]:
    """Torch side, in its own interpreter (main venv must stay torch-free)."""
    r = subprocess.run(
        [
            str(TORCH_PY),
            str(_HERE / "torch_runner.py"),
            case,
            str(inputs),
            weights,
            str(outputs),
        ],
        capture_output=True,
        text=True,
        cwd=_HERE,  # so torch_runner can `import cases`
    )
    if r.returncode != 0:
        raise RuntimeError(f"torch_runner failed:\n{r.stdout}\n{r.stderr}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def run_mlx(case: str, inputs: Path, weights: str, outputs: Path) -> dict[str, Any]:
    if str(_HERE) not in sys.path:  # work when invoked from any cwd
        sys.path.insert(0, str(_HERE))
    from mlx_runner import run_mlx as _run

    return _run(case, str(inputs), weights, str(outputs))


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
        if a.dtype.kind in "US":  # string payload (e.g. stack_keys): exact compare
            # 0-d vs 1-element str arrays are equivalent payloads; compare content.
            per[k] = {
                "shape": list(a.shape),
                "max_abs_diff": None,
                "bit_exact": bool(a.reshape(-1)[0] == b.reshape(-1)[0])
                if a.size == b.size == 1
                else bool(np.array_equal(a, b)),
            }
            continue
        if a.shape != b.shape:
            raise AssertionError(f"{k}: shape {a.shape} vs {b.shape}")
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        per[k] = {
            "shape": list(a.shape),
            "max_abs_diff": float(diff.max()) if a.size else 0.0,
            "bit_exact": bool(np.array_equal(a, b)),
        }
    return per


_META_ROWS = [
    ("torch_version", "torch version"),
    ("mlx_version", "mlx version"),
    ("torch_rmsnorm_eps_default", "torch RMSNorm eps default (probe, R5)"),
    ("effective_rmsnorm_eps", "effective RMSNorm eps (both sides)"),
    ("attention_path", "attention path"),
    ("device", "device"),
]


def write_report(
    case: str,
    seed: int,
    per: dict[str, dict[str, Any]],
    metas: list[dict[str, Any]],
    tol: float | None,
    require_bit_exact: bool,
    per_output_tol: dict[str, float | None] | None = None,
) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    suffix = "" if seed == 0 else f"_seed{seed}"
    path = REPORTS / f"{case}{suffix}.md"
    per_output_tol = per_output_tol or {}

    def _pass(k: str, v: dict[str, Any]) -> bool:
        if require_bit_exact:
            return bool(v["bit_exact"])
        gate = per_output_tol.get(k, tol)  # None entry ⇒ bit-exact for this key
        if k in per_output_tol and per_output_tol[k] is None:
            return bool(v["bit_exact"])
        return gate is not None and v["max_abs_diff"] is not None and v["max_abs_diff"] <= gate

    ok = all(_pass(k, v) for k, v in per.items())
    criterion = "bit-exact" if require_bit_exact else f"max abs diff <= {tol}"
    if per_output_tol:
        criterion += " (per-output overrides: " + ", ".join(
            f"{k}: {'bit-exact' if v is None else v}" for k, v in sorted(per_output_tol.items())
        ) + ")"
    lines = [
        f"# Parity report: `{case}`",
        "",
        f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- seed: {seed}",
        "- devices: torch=cpu, mlx=cpu (SPEC A1)",
        f"- criterion: {criterion}",
        f"- **verdict: {'PASS' if ok else 'FAIL'}**",
        "",
        "## Environment probe (SPEC R5)",
        "",
    ]
    meta = {k: v for m in metas for k, v in m.items()}
    lines += ["| key | value |", "|---|---|"]
    for key, label in _META_ROWS:
        if key in meta:
            lines.append(f"| {label} | {meta[key]} |")
    lines += [
        "",
        "## Outputs",
        "",
        "| output | shape | bit-exact | max abs diff |",
        "|---|---|---|---|",
    ]
    for k in sorted(per):
        v = per[k]
        diff = "n/a (exact compare)" if v["max_abs_diff"] is None else f"{v['max_abs_diff']:g}"
        lines.append(f"| `{k}` | {tuple(v['shape'])} | {v['bit_exact']} | {diff} |")
    lines.append("")
    path.write_text("\n".join(lines))
    if not ok:
        raise AssertionError(f"parity FAILED for {case}; report at {path}")
    return path


def run_parity_case(case: str, seed: int = 0) -> dict[str, Any]:
    """Full pipeline for one case; raises AssertionError on parity failure."""
    spec = CASES[case]
    inputs, weights = gen_case(case, seed)
    t_out = ARTIFACTS / f"{case}_seed{seed}_torch.npz"
    m_out = ARTIFACTS / f"{case}_seed{seed}_mlx.npz"
    t_meta = run_torch(case, inputs, weights, t_out)["meta"]
    m_meta = run_mlx(case, inputs, weights, m_out)["meta"]
    per = compare(t_out, m_out)
    report = write_report(
        case, seed, per, [t_meta, m_meta], spec.tol, spec.bit_exact, spec.per_output_tol
    )
    return {"per": per, "report": report}
