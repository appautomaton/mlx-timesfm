"""A1 parity for every operator ported in Phase 1b/1c (both stacks on CPU)."""

import pytest

from cases import CASES

pytestmark = pytest.mark.parity

OP_CASES = [
    "rmsnorm",
    "per_dim_scale",
    "rope_3d",
    "rope_4d",
    "attn_mask",
    "attn_mask_nc",
    "segment_mask",
    "mha_seq",
    "mha_var",
]


@pytest.mark.parametrize("case", OP_CASES)
def test_op_parity(case: str) -> None:
    from bridge import run_parity_case

    assert case in CASES
    run_parity_case(case, seed=0)
