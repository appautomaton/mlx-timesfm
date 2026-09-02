"""Guard the permanent MLX-only dependency and test boundary."""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_project_dependencies_exclude_retired_frameworks() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    requirements = list(project["project"].get("dependencies", ()))
    for group in project.get("dependency-groups", {}).values():
        requirements.extend(group)
    names = {
        re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].lower()
        for requirement in requirements
    }
    assert names.isdisjoint({"torch", "safetensors"})

    lock = (ROOT / "uv.lock").read_text()
    assert not re.search(r'^name = "(?:torch|safetensors)"$', lock, re.MULTILINE)


def test_package_and_tests_do_not_import_retired_frameworks() -> None:
    violations: list[str] = []
    retired = {"torch", "safetensors"}
    for base in (ROOT / "src", ROOT / "tests"):
        for path in base.rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    continue
                if any(module.split(".", 1)[0] in retired for module in modules):
                    violations.append(str(path.relative_to(ROOT)))
    assert not violations, f"retired framework imports found: {violations}"


def test_retired_live_parity_paths_are_absent() -> None:
    assert not (ROOT / ".venv-torch").exists()
    assert not (ROOT / "tests" / "parity").exists()
