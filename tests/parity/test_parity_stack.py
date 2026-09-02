"""A1 parity for Phase 2 stack structures (both stacks on CPU, random weights)."""

import pytest

pytestmark = pytest.mark.parity


def test_mixing_layer_parity() -> None:
    """One full MixingTransformer at REAL checkpoint dims (d=1280, hd=80)."""
    from bridge import run_parity_case

    run_parity_case("mixing_layer", seed=0)


def test_stacked_small_parity() -> None:
    """All 20 layers instantiated (small dims): residual chain, ×20 structure."""
    from bridge import run_parity_case

    run_parity_case("stacked_small", seed=0)


def test_stack_key_tree_matches_reference() -> None:
    """R6 at stack level: our 20-layer real-dim attribute tree must equal the
    reference StackedMixingTransformer state_dict key/shape set, exactly."""
    from pathlib import Path

    from bridge import ARTIFACTS, compare, gen_case, run_mlx, run_torch

    inputs, weights = gen_case("stack_keys", seed=0)
    t_out = Path(ARTIFACTS) / "stack_keys_seed0_torch.npz"
    m_out = Path(ARTIFACTS) / "stack_keys_seed0_mlx.npz"
    run_torch("stack_keys", inputs, weights, t_out)
    run_mlx("stack_keys", inputs, weights, m_out)
    per = compare(t_out, m_out)
    assert per["keys"]["bit_exact"], "state-dict key trees differ"
