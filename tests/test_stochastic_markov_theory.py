import numpy as np
import pytest

from stochx.stochastic import MarkovChain


def test_markov_return_and_stationary_helpers():
    chain = MarkovChain([[0.0, 1.0], [1.0, 0.0]])
    assert chain.first_return_probability(0, 1) == pytest.approx(0.0)
    assert chain.first_return_probability(0, 2) == pytest.approx(1.0)
    assert chain.return_probability(0) == 1.0
    assert chain.mean_return_time(0) == pytest.approx(2.0)


def test_reducible_stationary_distributions():
    chain = MarkovChain([[1.0, 0.0], [0.0, 1.0]])
    laws = chain.stationary_distributions()
    actual = {tuple(np.round(law, 12)) for law in laws}
    assert actual == {(1.0, 0.0), (0.0, 1.0)}
