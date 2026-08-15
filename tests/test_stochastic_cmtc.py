import numpy as np
import pytest

from optiflowx.stochastic import BirthDeathProcess, ContinuousTimeMarkovChain
from optiflowx.stochastic.exceptions import GeneratorValidationError, NumericalError


def test_generator_and_transition_matrix():
    chain = ContinuousTimeMarkovChain([[-2.0, 2.0], [1.0, -1.0]])
    np.testing.assert_allclose(chain.transition_matrix(0.0), np.eye(2))
    p = chain.transition_matrix(0.5)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert np.all(p >= -1e-12)


def test_kolmogorov_and_stationarity():
    chain = ContinuousTimeMarkovChain([[-2.0, 2.0], [1.0, -1.0]])
    pi = chain.stationary_distribution()
    np.testing.assert_allclose(pi @ chain.generator_matrix, [0.0, 0.0])
    np.testing.assert_allclose(chain.chapman_kolmogorov(0.2, 0.3), chain.transition_matrix(0.5))
    np.testing.assert_allclose(chain.forward_derivative(0.4), chain.backward_derivative(0.4))


def test_uniformization_matches_matrix_exponential():
    chain = ContinuousTimeMarkovChain([[-2.0, 2.0], [1.0, -1.0]])
    expected = chain.transition_matrix(1.75, method="expm")
    actual = chain.transition_matrix(1.75, method="uniformization")
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(actual.sum(axis=1), 1.0)
    assert np.all(actual >= -1e-12)


def test_uniformization_handles_higher_exit_rates():
    chain = ContinuousTimeMarkovChain([[-1000.0, 1000.0], [2.0, -2.0]])
    expected = chain.transition_matrix(0.5, method="expm")
    actual = chain.transition_matrix_uniformized(0.5, max_terms=2_000)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-11)


def test_uniformization_rejects_underprovisioned_truncation():
    chain = ContinuousTimeMarkovChain([[-1000.0, 1000.0], [2.0, -2.0]])
    with pytest.raises(NumericalError):
        chain.transition_matrix_uniformized(0.5, max_terms=10)


def test_invalid_generator():
    with pytest.raises(GeneratorValidationError):
        ContinuousTimeMarkovChain([[1.0, -1.0], [0.0, 0.0]])


def test_jump_chain_and_simulation():
    chain = ContinuousTimeMarkovChain(
        [[-2.0, 2.0, 0.0], [1.0, -3.0, 2.0], [0.0, 0.0, 0.0]],
        states=["A", "B", "C"],
    )
    jump = chain.jump_chain_matrix()
    np.testing.assert_allclose(jump[0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(jump[2], [0.0, 0.0, 1.0])
    path = chain.simulate(2.0, initial_state="A", rng=np.random.default_rng(123))
    assert path.times[0] == 0.0
    assert path.states[0] == "A"


def test_birth_death_process():
    process = BirthDeathProcess.finite([2.0, 3.0, 0.0], [0.0, 1.0, 4.0])
    q = process.generator_matrix()
    np.testing.assert_allclose(q.sum(axis=1), 0.0)
    np.testing.assert_allclose(process.jump_chain_matrix()[0, 1], 1.0)
    np.testing.assert_allclose(process.kolmogorov_derivative([1.0, 0.0, 0.0]), [-2.0, 2.0, 0.0])


def test_birth_death_stationary_product_formula_and_special_cases():
    process = BirthDeathProcess(lambda _k: 1.0, lambda k: max(1, k))
    np.testing.assert_allclose(process.stationary_weights(3), [1.0, 1.0, 0.5, 1 / 6])
    assert BirthDeathProcess.pure_birth_probability(2, 1.0, rate=0.5) > 0
    assert BirthDeathProcess.pure_death_probability(2, 1.0, initial_population=3, rate=0.5) > 0
    assert BirthDeathProcess.pure_immigration_probability(2, 1.0, rate=0.5) > 0
