"""Bridge end-to-end smoke: one trivial op round-trips through both stacks.

relu is checked bit-exact — elementwise max(x, 0) on fp32 CPU must not differ
between torch and mlx by even one ulp; if it does, the bridge itself is broken.
"""

import pytest

pytestmark = pytest.mark.parity


def test_relu_bit_exact() -> None:
    from bridge import run_parity_case

    result = run_parity_case("relu", seed=0, require_bit_exact=True)
    assert result["per"]["y"]["bit_exact"]
    assert result["report"].exists()
