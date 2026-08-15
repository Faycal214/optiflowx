import numpy as np
import pytest

from stochx.stochastic import CTMCPath, ContinuousTimeMarkovChain


def test_ctmc_domain_helpers():
    chain = ContinuousTimeMarkovChain([[-2.0, 2.0], [1.0, -1.0]])
    assert chain.communicating_classes() == [(0, 1)]
    np.testing.assert_allclose(chain.stationary_distribution_from_jump_chain(), [1 / 3, 2 / 3])
    assert chain.mean_return_time(0) == pytest.approx(1.5)


def test_ctmc_path_occupation_helpers():
    path = CTMCPath(np.asarray([0.0, 1.0, 3.0]), ("A", "B", "A"))
    assert path.occupation_time("A", 5.0) == pytest.approx(3.0)
    assert path.occupation_fraction("A", 5.0) == pytest.approx(0.6)
