"""Regression tests for centralized DTMC matrix validation."""

import numpy as np
import pytest

from stochx.stochastic import MarkovChain
from stochx.stochastic.exceptions import MatrixValidationError


def test_markov_chain_rejects_non_stochastic_matrix_with_semantic_error():
    with pytest.raises(MatrixValidationError):
        MarkovChain([[0.8, 0.3], [0.2, 0.8]])


def test_markov_chain_rejects_negative_probability_beyond_tolerance():
    with pytest.raises(MatrixValidationError):
        MarkovChain([[1.0 + 2e-12, -2e-12], [0.0, 1.0]])


def test_markov_chain_normalizes_row_sum_drift_on_construction():
    chain = MarkovChain(
        [[0.5, 0.5000000000005], [0.25, 0.7499999999995]],
    )
    np.testing.assert_allclose(
        chain.transition_matrix,
        [[0.5, 0.5], [0.25, 0.75]],
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        chain.transition_matrix.sum(axis=1),
        1.0,
        atol=0.0,
        rtol=0.0,
    )


def test_markov_chain_preserves_state_labels_and_tolerance():
    chain = MarkovChain(
        [[1.0, 0.0], [0.0, 1.0]],
        states=["healthy", "dead"],
        tolerance=1e-10,
    )
    assert chain.states == ("healthy", "dead")
