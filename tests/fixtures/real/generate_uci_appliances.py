#!/usr/bin/env python
"""Extract the compact real-data parity fixture from the UCI source archive.

Download the archive separately, then run:

    python tests/fixtures/real/generate_uci_appliances.py /path/to/archive.zip

The checked hash makes the extraction deterministic and guards against silent
upstream replacement.  Only the final 640 observations and the seven model
input columns are retained; the original archive is never stored in the repo.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import zipfile
from pathlib import Path

SOURCE_URL = (
    "https://archive.ics.uci.edu/static/public/374/"
    "appliances+energy+prediction.zip"
)
SOURCE_SHA256 = "2fccf354445d886e7917620b0195db1f3e3e34d5a067a93b844694a4c561255a"
MEMBER = "energydata_complete.csv"
NUM_SOURCE_ROWS = 19_735
NUM_FIXTURE_ROWS = 640  # context 512 + horizon 128
FIELDS = (
    "date",
    "Appliances",
    "T1",
    "RH_1",
    "T2",
    "RH_2",
    "T_out",
    "Press_mm_hg",
)
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "uci_appliances.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(archive: Path, output: Path = OUTPUT) -> None:
    actual_hash = _sha256(archive)
    if actual_hash != SOURCE_SHA256:
        raise ValueError(
            f"source SHA-256 mismatch: expected {SOURCE_SHA256}, got {actual_hash}"
        )

    with zipfile.ZipFile(archive) as zf, zf.open(MEMBER) as raw:
        rows = list(csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8")))
    if len(rows) != NUM_SOURCE_ROWS:
        raise ValueError(f"expected {NUM_SOURCE_ROWS} source rows, got {len(rows)}")

    selected = rows[-NUM_FIXTURE_ROWS:]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in selected:
            writer.writerow({field: row[field].strip() for field in FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    generate(args.archive, args.output)
    print(f"wrote {args.output} ({NUM_FIXTURE_ROWS} rows) from {SOURCE_URL}")


if __name__ == "__main__":
    main()
