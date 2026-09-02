#!/usr/bin/env python3
"""Inspect wheel/sdist metadata and reject private or oversized contents."""

from __future__ import annotations

import argparse
import email.parser
import tarfile
import zipfile
from pathlib import Path

EXPECTED_NAME = "mlx-timesfm"
EXPECTED_VERSION = "0.1.0"
FORBIDDEN_PARTS = {
    ".agents",
    ".git",
    ".references",
    ".venv",
    "models",
    "weights",
    "__pycache__",
}
FORBIDDEN_SUFFIXES = {".ckpt", ".gguf", ".npy", ".npz", ".onnx", ".pt", ".pth", ".safetensors"}


def _check_names(names: list[str], *, wheel: bool) -> None:
    for name in names:
        path = Path(name)
        if any(part in FORBIDDEN_PARTS for part in path.parts):
            raise ValueError(f"forbidden distribution path: {name}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ValueError(f"forbidden distribution asset: {name}")
        if wheel and "tests" in path.parts:
            raise ValueError(f"tests must not be included in the wheel: {name}")


def _check_metadata(raw: bytes) -> None:
    metadata = email.parser.BytesParser().parsebytes(raw)
    if metadata["Name"] != EXPECTED_NAME:
        raise ValueError(f"unexpected metadata name: {metadata['Name']!r}")
    if metadata["Version"] != EXPECTED_VERSION:
        raise ValueError(f"unexpected metadata version: {metadata['Version']!r}")
    requires_python = (metadata["Requires-Python"] or "").replace(" ", "")
    if requires_python != ">=3.13,<3.14":
        raise ValueError(f"unexpected Requires-Python: {metadata['Requires-Python']!r}")
    requirements = metadata.get_all("Requires-Dist", ())
    normalized_requirements = [requirement.replace(" ", "") for requirement in requirements]
    if normalized_requirements != ["mlx>=0.32.2,<0.33"]:
        raise ValueError(f"unexpected runtime requirements: {requirements}")
    if metadata["License-Expression"] != "Apache-2.0":
        raise ValueError(f"unexpected license expression: {metadata['License-Expression']!r}")


def check_wheel(path: Path) -> None:
    if path.stat().st_size > 500_000:
        raise ValueError(f"wheel is unexpectedly large: {path.stat().st_size} bytes")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        _check_names(names, wheel=True)
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        _check_metadata(archive.read(metadata_name))
        if not any(name.endswith("mlx_timesfm/py.typed") for name in names):
            raise ValueError("wheel does not include py.typed")


def check_sdist(path: Path) -> None:
    if path.stat().st_size > 1_000_000:
        raise ValueError(f"sdist is unexpectedly large: {path.stat().st_size} bytes")
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
        _check_names(names, wheel=False)
        required_suffixes = {
            "/scripts/check_release.py",
            "/tests/test_release.py",
            "/uv.lock",
        }
        missing = {
            suffix
            for suffix in required_suffixes
            if not any(name.endswith(suffix) for name in names)
        }
        if missing:
            raise ValueError(f"sdist is missing release sources: {sorted(missing)}")
        pkg_info = next(
            member for member in archive.getmembers() if member.name.endswith("/PKG-INFO")
        )
        extracted = archive.extractfile(pkg_info)
        if extracted is None:
            raise ValueError("could not read sdist PKG-INFO")
        _check_metadata(extracted.read())


def check(path: Path) -> None:
    if path.suffix == ".whl":
        check_wheel(path)
    elif path.name.endswith(".tar.gz"):
        check_sdist(path)
    else:
        raise ValueError(f"unsupported distribution: {path}")
    print(f"validated {path} ({path.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()
    for artifact in args.artifacts:
        check(artifact)


if __name__ == "__main__":
    main()
