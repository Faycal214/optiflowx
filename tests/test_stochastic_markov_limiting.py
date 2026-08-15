import numpy as np

from stochx.stochastic import MarkovChain


def test_limiting_distribution_single_closed_three_state_class():
    chain = MarkovChain(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.5, 0.0, 0.5],
        ],
        states=["T", "A", "B", "C"],
    )

    expected = np.tile([0.0, 1 / 3, 1 / 3, 1 / 3], (4, 1))
    np.testing.assert_allclose(chain.limiting_distribution(), expected)
