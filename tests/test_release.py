"""Release metadata stays internally consistent."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_release_source_metadata() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_release.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "mlx-timesfm 0.1.0 source metadata valid"
