"""Bridge machinery that does NOT need torch (runs in the A4 torch-free suite)."""

import numpy as np

from bridge import gen_case, run_mlx


def test_main_process_stays_torch_free() -> None:
    # Review #1 / N1: the bridge talks to torch ONLY via the .venv-torch
    # subprocess; nothing on this side may ever pull torch into sys.modules.
    import sys

    assert "torch" not in sys.modules


def test_gen_case_is_deterministic(tmp_path) -> None:
    a_path, _ = gen_case("relu", seed=7, out_dir=tmp_path)
    b_path, _ = gen_case("relu", seed=7, out_dir=tmp_path)
    a, b = np.load(a_path), np.load(b_path)
    for k in a.files:
        assert np.array_equal(a[k], b[k])
    c_path, _ = gen_case("relu", seed=8, out_dir=tmp_path)
    c = np.load(c_path)
    assert not np.array_equal(a["x"], c["x"])


def test_mlx_runner_runs_and_stamps_meta(tmp_path) -> None:
    inputs, weights = gen_case("relu", seed=1, out_dir=tmp_path)
    outputs = tmp_path / "out.npz"
    info = run_mlx("relu", inputs, weights, outputs)
    assert info["meta"]["device"] == "cpu"
    assert info["meta"]["effective_rmsnorm_eps"] == 1e-5
    out = np.load(outputs)
    assert np.all(out["y"] >= 0)


def test_parity_eps_pinned_to_inference_eps() -> None:
    # cases.py can't import mlx_timesfm (torch-env shareable); pin them equal here.
    from cases import PARITY_EPS
    from mlx_timesfm.config import INFERENCE_RMSNORM_EPS

    assert PARITY_EPS == INFERENCE_RMSNORM_EPS
