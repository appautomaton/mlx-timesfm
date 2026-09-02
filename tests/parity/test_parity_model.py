"""A1 parity for Phase 3 full-sequence forward (random weights, ≥2 seeds).

Compares logits + revin stats + aux outputs (resblock input, transformer
input/output, per-layer seq-attn masks) across the two stacks. Real-weights
e2e intentionally NOT here (held for review; PLAN Phase 5 gate).
"""

import pytest

pytestmark = pytest.mark.parity

SEEDS = (0, 1)


@pytest.mark.parametrize("case", ["model_forward_small", "model_forward_freeze"])
@pytest.mark.parametrize("seed", SEEDS)
def test_model_forward_parity(case: str, seed: int) -> None:
    from bridge import run_parity_case

    run_parity_case(case, seed=seed)
