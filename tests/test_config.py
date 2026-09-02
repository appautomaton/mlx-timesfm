"""config.py tests — runnable without torch and without the weights dir.

Review #2: the committed `tests/fixtures/config.json` (small, synthetic,
checkpoint-shaped) is the default test path; the real `models/` symlink is
gitignored and must never be required by the suite (A4).
"""

import json
from pathlib import Path

import pytest

from mlx_timesfm.config import (
    INFERENCE_RMSNORM_EPS,
    ResidualBlockConfig,
    StackedTransformersConfig,
    TimesFM3Config,
    TransformerConfig,
    load_config,
)

REPO = Path(__file__).resolve().parents[1]
FIXTURE_JSON = REPO / "tests/fixtures/config.json"
CKPT_DIR = REPO / "models/timesfm_3_0/original"


def test_default_config_is_the_checkpoint_shape() -> None:
    cfg = TimesFM3Config()
    assert cfg.rolls == 2
    assert cfg.resblock_input_dim == 192  # 2*(32+64), PLAN cheat-sheet
    assert cfg.num_quantiles == 9
    assert cfg.transformer_config.num_layers == 20


def test_rmsnorm_eps_unset_at_dataclass_level() -> None:
    # SPEC R5: from_dict/dataclass never invents an eps.
    assert TimesFM3Config().rmsnorm_eps is None
    assert TimesFM3Config.from_dict(json.loads(FIXTURE_JSON.read_text())).rmsnorm_eps is None


def test_load_config_applies_documented_inference_eps() -> None:
    # Review #3: the inference boundary hard-wires a documented eps...
    assert load_config(FIXTURE_JSON).rmsnorm_eps == INFERENCE_RMSNORM_EPS
    # ...and parity overrides it explicitly.
    assert load_config(FIXTURE_JSON, rmsnorm_eps=1e-6).rmsnorm_eps == 1e-6


def test_to_dict_is_checkpoint_shaped() -> None:
    # Review #4: no port-only knobs leak into checkpoint export.
    raw = json.loads(FIXTURE_JSON.read_text())
    cfg = TimesFM3Config.from_dict(raw)
    assert "rmsnorm_eps" not in cfg.to_dict()
    assert set(raw) == set(cfg.to_dict())
    assert cfg.to_dict() == raw  # full round-trip against checkpoint-shaped json


def test_validation_matches_reference_model_init() -> None:
    with pytest.raises(ValueError, match="multiple"):
        TimesFM3Config(input_patch_len=32, output_patch_len=100)
    with pytest.raises(ValueError, match="output_dims"):
        TimesFM3Config(
            residual_block_config=ResidualBlockConfig(
                hidden_dims=1280, output_dims=640, use_bias=False, activation="relu"
            )
        )
    with pytest.raises(ValueError, match="stitching"):
        TimesFM3Config(input_patch_len=64, output_patch_len=64)


def _minimal_dict() -> dict:
    return TimesFM3Config(
        transformer_config=StackedTransformersConfig(
            num_layers=2,
            transformer=TransformerConfig(
                model_dims=32,
                hidden_dims=64,
                num_heads=4,
                attention_norm="rms",
                feedforward_norm="rms",
                qk_norm="rms",
                use_bias=False,
                use_rope_seq=True,
                use_rope_var=False,
                ff_activation="relu",
                deterministic=True,
            ),
        ),
        residual_block_config=ResidualBlockConfig(
            hidden_dims=64, output_dims=32, use_bias=False, activation="relu"
        ),
    ).to_dict()


def test_dict_round_trip() -> None:
    d = _minimal_dict()
    assert TimesFM3Config.from_dict(d).to_dict() == d


@pytest.mark.skipif(not CKPT_DIR.exists(), reason="weights dir absent (clean clone)")
def test_real_checkpoint_config_parses() -> None:
    raw = json.loads((CKPT_DIR / "config.json").read_text())
    cfg = TimesFM3Config.from_dict(raw)  # faithful parse, no eps injection
    assert cfg.transformer_config.num_layers == 20
    assert cfg.transformer_config.transformer.model_dims == 1280
    assert cfg.transformer_config.transformer.num_heads == 16
    assert cfg.resblock_input_dim == 192
    assert cfg.input_transform == "identity"
    assert cfg.rmsnorm_eps is None
    assert cfg.to_dict() == raw  # official config.json round-trips exactly
