"""Load the compact UCI real-data golden regression case."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "real" / "uci_appliances.csv"
CONTEXT = 512
HORIZON = 128
TARGET_COLUMNS = ("Appliances", "T1", "RH_1")
PAST_ONLY_COLUMNS = ("T2", "RH_2")
PAST_FUTURE_COLUMNS = ("T_out", "Press_mm_hg")


def load_case(path: Path = FIXTURE) -> dict[str, np.ndarray]:
    """Return b=1 float32 arrays for the golden inference regression."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    expected = CONTEXT + HORIZON
    if len(rows) != expected:
        raise ValueError(f"expected {expected} fixture rows, got {len(rows)}")

    def values(columns: tuple[str, ...]) -> np.ndarray:
        data = np.array(
            [[float(row[column]) for row in rows] for column in columns],
            dtype=np.float32,
        )
        return np.ascontiguousarray(data[None])

    targets = values(TARGET_COLUMNS)
    past_only = values(PAST_ONLY_COLUMNS)
    past_future = values(PAST_FUTURE_COLUMNS)
    return {
        "target": targets[..., :CONTEXT],
        "past_only": past_only[..., :CONTEXT],
        "past_future": past_future,
        "truth": targets[..., CONTEXT:],
        "target_sigma": targets[..., :CONTEXT].std(axis=-1, dtype=np.float64)[0],
    }
