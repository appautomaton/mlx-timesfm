"""Bridge machinery that does NOT need torch (runs in the A4 torch-free suite)."""

import numpy as np
from bridge import gen_case, run_mlx


def test_gen_case_is_deterministic() -> None:
    a = np.load(gen_case("relu", seed=7))
    b = np.load(gen_case("relu", seed=7))
    for k in a.files:
        assert np.array_equal(a[k], b[k])
    c = np.load(gen_case("relu", seed=8))
    assert not np.array_equal(a["x"], c["x"])


def test_main_process_stays_torch_free() -> None:
    # Review #1 / N1: the bridge talks to torch ONLY via the .venv-torch
    # subprocess; nothing on this side may ever pull torch into sys.modules.
    import sys

    assert "torch" not in sys.modules


def test_mlx_runner_runs_and_stamps_meta(tmp_path) -> None:
    inputs = gen_case("relu", seed=1, out_dir=tmp_path)
    outputs = tmp_path / "out.npz"
    info = run_mlx("relu", inputs, outputs)
    assert info["meta"]["device"] == "cpu"
    out = np.load(outputs)
    assert np.all(out["y"] >= 0)
