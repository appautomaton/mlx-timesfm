"""Pytest setup.

`parity`-marked tests need the torch reference env (`.venv-torch/`, gitignored).
SPEC A4: the default suite must stay green in a clean torch-free clone, so those
tests skip cleanly instead of erroring when the env is absent.
"""

from pathlib import Path

import pytest

TORCH_PY = Path(__file__).resolve().parents[1] / ".venv-torch" / "bin" / "python"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "parity: torch↔mlx parity test (needs .venv-torch/; auto-skips without it)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if TORCH_PY.exists():
        return
    skip = pytest.mark.skip(reason=".venv-torch/ absent (SPEC A4: skip, don't fail)")
    for item in items:
        # get_closest_marker, not `"parity" in item.keywords` — the directory
        # name tests/parity/ would make every path beneath it match by accident.
        if item.get_closest_marker("parity"):
            item.add_marker(skip)
