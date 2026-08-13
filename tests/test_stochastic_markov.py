import numpy as np
import pytest

from optiflowx.stochastic import MarkovChain


def test_transition_matrix_is_validated_and_exposed():
    chain = MarkovChain([[0.7, 0.3], [0.4, 0.6]], states=["A", "B"])

    np.testing.assert_allclose(chain.transition_matrix, [[0.7, 0.3], [0.4, 0.6]])
    assert chain.states == ("A", "B")
    assert chain.n_states == 2


def test_n_step_transition():
    chain = MarkovChain([[0.7, 0.3], [0.4, 0.6]])
    np.testing.assert_allclose(
        chain.n_step_transition(2),
        np.linalg.matrix_power(chain.transition_matrix, 2),
    )


def test_stationary_distribution():
    chain = MarkovChain([[0.7, 0.3], [0.4, 0.6]])
    stationary = chain.stationary_distribution()

    np.testing.assert_allclose(stationary, [4 / 7, 3 / 7])
    np.testing.assert_allclose(stationary @ chain.transition_matrix, stationary)


def test_classes_irreducibility_period_and_classification():
    chain = MarkovChain(
        [
            [0.7, 0.3, 0.0],
            [0.4, 0.6, 0.0],
            [0.0, 0.0, 1.0],
        ],
        states=["A", "B", "C"],
    )

    assert not chain.is_irreducible()
    assert chain.is_aperiodic()
    assert set(chain.classify_states()) == {"A", "B", "C"}
    assert chain.classify_states()["A"] == "recurrent"
    assert chain.classify_states()["B"] == "recurrent"
    assert chain.classify_states()["C"] == "recurrent"


def test_simulation_is_reproducible_with_generator():
    chain = MarkovChain([[0.7, 0.3], [0.4, 0.6]], states=["A", "B"])
    first = chain.simulate(10, initial_state="A", rng=np.random.default_rng(123))
    second = chain.simulate(10, initial_state="A", rng=np.random.default_rng(123))

    assert first == second
    assert len(first) == 11


def test_invalid_transition_matrix_is_rejected():
    with pytest.raises(ValueError):
        MarkovChain([[0.2, 0.2], [0.5, 0.5]])

    with pytest.raises(ValueError):
        MarkovChain([[0.7, -0.3], [0.4, 0.6]])


def test_reducible_chain_requires_explicit_closed_class_for_stationarity():
    chain = MarkovChain([[1.0, 0.0], [0.2, 0.8]])

    with pytest.raises(ValueError, match="reducible"):
        chain.stationary_distribution()
