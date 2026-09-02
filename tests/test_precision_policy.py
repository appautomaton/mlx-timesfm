"""Process-level precision policy for the fp32 inference baseline."""

from __future__ import annotations

import os
import subprocess
import sys


def _run_with(value: str | None) -> str:
    env = os.environ.copy()
    if value is None:
        env.pop("MLX_ENABLE_TF32", None)
    else:
        env["MLX_ENABLE_TF32"] = value
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, mlx_timesfm; print(os.environ['MLX_ENABLE_TF32'])",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout.strip()


def test_full_fp32_is_the_default() -> None:
    assert _run_with(None) == "0"


def test_explicit_precision_override_is_preserved() -> None:
    assert _run_with("1") == "1"
