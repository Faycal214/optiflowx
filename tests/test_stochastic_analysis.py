import numpy as np

from optiflowx.stochastic.analysis import empirical_state_frequencies


def test_empirical_state_frequencies():
    np.testing.assert_allclose(
        empirical_state_frequencies(["A", "A", "B", "A"], ["A", "B"]),
        [0.75, 0.25],
    )
