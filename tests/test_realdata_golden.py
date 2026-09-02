"""MLX regression against the original-reference real-data golden."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from realdata_case import HORIZON, TARGET_COLUMNS, load_case

from mlx_timesfm import load

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "tests" / "fixtures" / "real"
MANIFEST = REAL / "golden_manifest.json"
CHECKPOINT = ROOT / "models" / "timesfm_3_0" / "original"
REQUIRES_CHECKPOINT = pytest.mark.skipif(
    not (CHECKPOINT / "model.safetensors").is_file(),
    reason="real checkpoint absent (models/ is a gitignored symlink)",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _golden() -> np.ndarray:
    rows = list(csv.DictReader((REAL / "uci_appliances_golden.csv").open()))
    result = np.empty((1, len(TARGET_COLUMNS), HORIZON, 9), dtype=np.float32)
    for target_index, target in enumerate(TARGET_COLUMNS):
        selected = [row for row in rows if row["target"] == target]
        if len(selected) != HORIZON:
            raise ValueError(f"expected {HORIZON} golden rows for {target}")
        for step_index, row in enumerate(selected):
            if int(row["step"]) != step_index + 1:
                raise ValueError(f"non-contiguous golden steps for {target}")
            result[0, target_index, step_index] = [
                float(row[f"q{quantile}"]) for quantile in range(10, 100, 10)
            ]
    return result


def test_realdata_artifact_hashes() -> None:
    manifest = json.loads(MANIFEST.read_text())
    assert _sha256(REAL / manifest["fixture"]["file"]) == manifest["fixture"]["sha256"]
    assert _sha256(REAL / manifest["golden"]["file"]) == manifest["golden"]["sha256"]


@REQUIRES_CHECKPOINT
def test_real_checkpoint_hashes() -> None:
    manifest = json.loads(MANIFEST.read_text())
    assert _sha256(CHECKPOINT / "model.safetensors") == manifest["reference"]["checkpoint_sha256"]
    assert _sha256(CHECKPOINT / "config.json") == manifest["reference"]["config_sha256"]


@pytest.mark.parametrize(
    ("device_name", "device"),
    (("cpu", mx.cpu), ("gpu", mx.gpu)),
)
@REQUIRES_CHECKPOINT
def test_realdata_golden_forecast(device_name: str, device: mx.Device) -> None:
    if device_name == "gpu" and os.environ.get("MLX_ENABLE_TF32") != "0":
        pytest.fail("golden GPU parity requires MLX_ENABLE_TF32=0")

    case = load_case()
    expected = _golden()
    previous = mx.default_device()
    mx.set_default_device(device)
    try:
        model = load(CHECKPOINT)
        actual = model.decode(
            mx.array(case["target"]),
            horizon=HORIZON,
            past_only_covariates=mx.array(case["past_only"]),
            past_future_covariates=mx.array(case["past_future"]),
        )[:, : len(TARGET_COLUMNS)]
        mx.eval(actual)
        difference = np.abs(np.asarray(actual).astype(np.float64) - expected)
    finally:
        mx.set_default_device(previous)

    per_target = difference.max(axis=(0, 2, 3))
    tolerance = 2e-3 * case["target_sigma"]
    assert np.all(per_target <= tolerance), {
        target: {"max_abs": float(error), "tolerance": float(limit)}
        for target, error, limit in zip(TARGET_COLUMNS, per_target, tolerance, strict=True)
    }

    actual_np = np.asarray(actual)
    sigma = case["target_sigma"][None, :, None]
    beyond_band = 0
    for quantile in range(expected.shape[-1] - 1):
        reference_gap = expected[..., quantile + 1] - expected[..., quantile]
        candidate_gap = actual_np[..., quantile + 1] - actual_np[..., quantile]
        new_crossing = (candidate_gap < 0.0) & (reference_gap >= 0.0)
        beyond_band += int((new_crossing & (reference_gap > 4e-3 * sigma)).sum())
    assert beyond_band == 0
