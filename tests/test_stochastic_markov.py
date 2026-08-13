import numpy as np
import pytest

from optiflowx.stochastic import MarkovChain


def test_transition_matrix_is_validated_and_exposed():
    chain = MarkovChain([[0.7, 0.3], [0.4, 0.6]], states=["A", "B"])

    np.testing.assert_allclose(chain.transition_matrix, [[0.7, 0.3], [0.4, 0.6]])
    assert chain.states == ("A", "B")
    assert chain.n_states == 2


def test_n_step_transition_and_chapman_kolmogorov():
    chain = MarkovChain([[0.7, 0.3], [0.4, 0.6]])
    expected = np.linalg.matrix_power(chain.transition_matrix, 5)

    np.testing.assert_allclose(chain.n_step_transition(5), expected)
    np.testing.assert_allclose(chain.chapman_kolmogorov(2, 3), expected)


def test_state_distribution():
    chain = MarkovChain([[0.7, 0.3], [0.4, 0.6]])
    mu0 = [1.0, 0.0]
    expected = np.asarray(mu0) @ np.linalg.matrix_power(chain.transition_matrix, 2)
    np.testing.assert_allclose(chain.state_distribution(mu0, 2), expected)


def test_accessibility_and_communication_classes():
    chain = MarkovChain(
        [[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        states=["A", "B", "C"],
    )

    assert chain.accessible("A", "B")
    assert not chain.accessible("B", "A")
    assert chain.communicate("B", "B")
    assert {frozenset(c) for c in chain.communicating_classes()} == {
        frozenset({"A"}), frozenset({"B"}), frozenset({"C"})
    }


def test_closed_and_absorbing_classes():
    chain = MarkovChain(
        [[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        states=["A", "B", "C"],
    )

    assert {frozenset(c) for c in chain.closed_classes()} == {
        frozenset({"B"}), frozenset({"C"})
    }
    assert chain.is_absorbing_state("B")
    assert not chain.is_absorbing_state("A")


def test_recurrence_transience():
    chain = MarkovChain(
        [[0.7, 0.3, 0.0], [0.4, 0.6, 0.0], [0.4, 0.0, 0.6]],
        states=["A", "B", "C"],
    )

    classifications = chain.classify_states()
    assert classifications["A"] == "recurrent"
    assert classifications["B"] == "recurrent"
    assert classifications["C"] == "transient"
    assert not chain.is_irreducible()
    assert not chain.is_ergodic()


def test_periodicity_and_ergodicity():
    periodic = MarkovChain([[0.0, 1.0], [1.0, 0.0]])
    assert periodic.period(0) == 2
    assert periodic.period(1) == 2
    assert not periodic.is_aperiodic()
    assert not periodic.is_ergodic()

    ergodic = MarkovChain([[0.7, 0.3], [0.4, 0.6]])
    assert ergodic.period(0) == 1
    assert ergodic.period(1) == 1
    assert ergodic.is_aperiodic()
    assert ergodic.is_ergodic()


def test_first_visit_probability_and_visit_probability():
    chain = MarkovChain([[0.5, 0.5], [0.0, 1.0]])

    assert chain.first_visit_probability(0, 1, 1) == pytest.approx(0.5)
    assert chain.first_visit_probability(0, 1, 2) == pytest.approx(0.25)
    assert chain.visit_probability(0, 1, 2) == pytest.approx(0.75)


def test_expected_hitting_time():
    chain = MarkovChain([[0.5, 0.5], [0.0, 1.0]])
    assert chain.expected_hitting_time(0, 1) == pytest.approx(2.0)
    assert chain.expected_hitting_time(1, 0) == float("inf")


def test_stationary_distribution():
    chain = MarkovChain([[0.7, 0.3], [0.4, 0.6]])
    stationary = chain.stationary_distribution()

    np.testing.assert_allclose(stationary, [4 / 7, 3 / 7])
    np.testing.assert_allclose(stationary @ chain.transition_matrix, stationary)


def test_limiting_distribution_for_ergodic_chain():
    chain = MarkovChain([[0.7, 0.3], [0.4, 0.6]])
    limit = chain.limiting_distribution()
    expected = np.tile([4 / 7, 3 / 7], (2, 1))
    np.testing.assert_allclose(limit, expected)


def test_periodic_chain_has_stationary_but_no_limit():
    chain = MarkovChain([[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(chain.stationary_distribution(), [0.5, 0.5])
    with pytest.raises(ValueError):
        chain.limiting_distribution()


def test_absorption_probability():
    chain = MarkovChain(
        [[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        states=["A", "B", "C"],
    )

    assert chain.absorption_probability("A", ["B"]) == pytest.approx(1.0)
    assert chain.absorption_probability("B", ["B"]) == pytest.approx(1.0)
    assert chain.absorption_probability("A", ["C"]) == pytest.approx(0.0)


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


def test_reducible_chain_requires_irreducibility_for_unique_stationarity():
    chain = MarkovChain([[1.0, 0.0], [0.2, 0.8]])
    with pytest.raises(ValueError, match="irreducible"):
        chain.stationary_distribution()
