"""A1 parity for Phase 3 full-sequence forward (random weights, ≥2 seeds)
plus the real-weights forward run (torch SDPA-off, forced-equal eps — PLAN
Phase 3 held item; NOT a claim of official-default-kernel parity).

Compares logits + revin stats + aux outputs (resblock input, transformer
input/output, per-layer seq-attn masks) across the two stacks, CPU vs CPU.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.parity

_ROOT = Path(__file__).resolve().parents[2]
_CKPT = _ROOT / "models" / "timesfm_3_0" / "original"

SEEDS = (0, 1)


@pytest.mark.parametrize("case", ["model_forward_small", "model_forward_freeze"])
@pytest.mark.parametrize("seed", SEEDS)
def test_model_forward_parity(case: str, seed: int) -> None:
    from bridge import run_parity_case

    run_parity_case(case, seed=seed)


@pytest.mark.skipif(
    not (_CKPT / "model.safetensors").is_file(),
    reason="real checkpoint absent (models/ is a gitignored symlink)",
)
def test_model_forward_real_weights_parity() -> None:
    from bridge import run_parity_case

    run_parity_case("model_forward_real", seed=0)
