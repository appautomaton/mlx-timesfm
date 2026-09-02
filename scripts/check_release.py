#!/usr/bin/env python3
"""Validate source-tree metadata before building or releasing."""

from __future__ import annotations

import argparse
import ast
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_NAME = "mlx-timesfm"
EXPECTED_REPOSITORY = "https://github.com/appautomaton/mlx-timesfm"
REQUIRED_FILES = (
    "CHANGELOG.md",
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "src/mlx_timesfm/py.typed",
)
RETIRED_IMPORTS = {"torch", "safetensors"}
SOURCE_NOTICE_MARKERS = (
    "Copyright 2026 Google LLC",
    "Modifications Copyright 2026 AppAutomaton",
)


def _dependency_name(requirement: str) -> str:
    return re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].strip().lower()


def _import_violations() -> list[str]:
    violations: list[str] = []
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
                if any(module.split(".", 1)[0] in RETIRED_IMPORTS for module in modules):
                    violations.append(str(path.relative_to(ROOT)))
    return sorted(set(violations))


def check(tag: str | None = None) -> str:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    name = project["name"]
    version = project["version"]
    if name != EXPECTED_NAME:
        raise ValueError(f"project name must be {EXPECTED_NAME!r}, got {name!r}")
    if project["urls"]["Repository"] != EXPECTED_REPOSITORY:
        raise ValueError("repository URL does not match the trusted-publisher identity")
    if project.get("license") != "Apache-2.0":
        raise ValueError("project license must use the Apache-2.0 SPDX expression")

    dependencies = project.get("dependencies", ())
    dependency_names = {_dependency_name(requirement) for requirement in dependencies}
    if dependency_names != {"mlx"}:
        raise ValueError(
            f"runtime dependency set must be exactly {{'mlx'}}, got {dependency_names}"
        )
    if violations := _import_violations():
        raise ValueError(f"retired framework imports found: {violations}")

    missing_notices = [
        str(path.relative_to(ROOT))
        for path in (ROOT / "src" / "mlx_timesfm").glob("*.py")
        if any(marker not in path.read_text() for marker in SOURCE_NOTICE_MARKERS)
    ]
    if missing_notices:
        raise ValueError(f"source attribution notices missing: {missing_notices}")

    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"required release files missing: {missing}")
    if len((ROOT / "README.md").read_text()) < 2_000:
        raise ValueError("README.md is unexpectedly short")

    expected_tag = f"v{version}"
    if tag is not None and tag != expected_tag:
        raise ValueError(f"release tag must be {expected_tag!r}, got {tag!r}")
    return f"{name} {version} source metadata valid"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag")
    args = parser.parse_args()
    print(check(args.tag))


if __name__ == "__main__":
    main()
