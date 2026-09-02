"""Real-checkpoint tests — skipif the (gitignored) weights dir is absent.

Locks SPEC R6 against the actual 445-tensor state dict: the module attribute
tree must produce EXACTLY the safetensors keys, shapes and dtypes, and
``load()`` must refuse partial loads. (The 445-way reconciliation was manual
until this file.)

Runs torch-free (A4): mx.load reads safetensors natively. The safetensors
HEADER is all we touch for the key-tree test — tensors stay lazily mapped.
"""

from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx_timesfm import TimesFM3, load, load_config

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "models" / "timesfm_3_0" / "original"
CONFIG_JSON = CKPT / "config.json"
WEIGHTS = CKPT / "model.safetensors"

pytestmark = pytest.mark.skipif(
    not (CONFIG_JSON.is_file() and WEIGHTS.is_file()),
    reason="real checkpoint absent (models/ is a gitignored symlink — SPEC A4)",
)

EXPECTED_NUM_TENSORS = 445
EXPECTED_NUM_PARAMS = 330_710_976


@pytest.fixture(scope="module")
def ckpt_params() -> dict[str, mx.array]:
    return mx.load(str(WEIGHTS))


def test_state_dict_key_tree_round_trip(ckpt_params: dict[str, mx.array]) -> None:
    """sorted(flatten(model.parameters())) == sorted(safetensors keys), with
    matching shapes AND dtypes (SPEC R6)."""
    model = TimesFM3(load_config(CKPT))
    ours = dict(tree_flatten(model.parameters()))

    assert len(ckpt_params) == EXPECTED_NUM_TENSORS
    assert sorted(ours) == sorted(ckpt_params)
    bad_shape = {
        k: (ours[k].shape, v.shape) for k, v in ckpt_params.items()
        if ours[k].shape != v.shape
    }
    bad_dtype = {
        k: (str(v.dtype)) for k, v in ckpt_params.items() if v.dtype != mx.float32
    }
    assert not bad_shape, f"shape mismatches: {list(bad_shape.items())[:5]}"
    assert not bad_dtype, f"non-fp32 tensors: {list(bad_dtype.items())[:5]}"
    assert sum(v.size for v in ckpt_params.values()) == EXPECTED_NUM_PARAMS


def test_load_full_checkpoint_strict() -> None:
    """load() succeeds on the real dir (strict R6 loader, nothing skipped)."""
    model = load(CKPT)
    params = dict(tree_flatten(model.parameters()))
    assert len(params) == EXPECTED_NUM_TENSORS
    sample = params["output_head.bias"]
    mx.eval(sample)
    assert sample.shape == (576,)


def test_load_rejects_partial_state_dict(tmp_path: Path, ckpt_params: dict) -> None:
    """R6: dropping a tensor or gifting an extra one must RAISE, not warn."""
    import shutil

    shutil.copy(CONFIG_JSON, tmp_path / "config.json")

    # write a safetensors copy with one key removed, via MLX native save
    trimmed = dict(ckpt_params)
    trimmed.pop("output_head.bias")
    mx.save_safetensors(str(tmp_path / "model.safetensors"), trimmed)
    with pytest.raises(ValueError, match="missing"):
        load(tmp_path)
