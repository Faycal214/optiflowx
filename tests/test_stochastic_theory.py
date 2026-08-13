import numpy as np
import pytest

from optiflowx.stochastic import (
    CTMCPath,
    ContinuousTimeMarkovChain,
    Filtration,
    FiniteProbabilitySpace,
    MarkovChain,
    Martingale,
    Partition,
    StoppingTime,
    conditional_characterization_error,
    conditional_expectation_given_event,
    conditional_probability_given_event,
    ctmc_communication_classes,
    ctmc_mean_return_time,
    ctmc_stationary_from_jump_chain,
    empirical_state_frequencies,
    first_return_probability,
    independent_random_variables,
    mean_return_time,
    occupation_fraction,
    occupation_time,
    return_probability,
    stationary_distributions,
    stopped_martingale,
    transform_martingale,
)


def test_first_return_and_mean_return_time():
    chain = MarkovChain([[0.0, 1.0], [1.0, 0.0]])
    assert first_return_probability(chain, 0, 1) == pytest.approx(0.0)
    assert first_return_probability(chain, 0, 2) == pytest.approx(1.0)
    assert return_probability(chain, 0) == 1.0
    assert mean_return_time(chain, 0) == pytest.approx(2.0)


def test_multiple_stationary_distributions_for_reducible_chain():
    chain = MarkovChain([[1.0, 0.0], [0.0, 1.0]])
    laws = stationary_distributions(chain)
    assert len(laws) == 2
    actual = {tuple(np.round(law, 12)) for law in laws}
    assert actual == {(1.0, 0.0), (0.0, 1.0)}


def test_empirical_state_frequencies():
    frequencies = empirical_state_frequencies(["A", "A", "B", "A"], ["A", "B"])
    np.testing.assert_allclose(frequencies, [0.75, 0.25])


def test_ctmc_jump_chain_stationary_relation_and_return_time():
    chain = ContinuousTimeMarkovChain([[-2.0, 2.0], [1.0, -1.0]])
    assert ctmc_communication_classes(chain) == [(0, 1)]
    np.testing.assert_allclose(ctmc_stationary_from_jump_chain(chain), [1 / 3, 2 / 3])
    assert ctmc_mean_return_time(chain, 0) == pytest.approx(1.5)


def test_ctmc_occupation_fraction():
    path = CTMCPath(np.asarray([0.0, 1.0, 3.0]), ("A", "B", "A"))
    assert occupation_time(path, "A", 5.0) == pytest.approx(3.0)
    assert occupation_fraction(path, "A", 5.0) == pytest.approx(0.6)


def test_conditional_expectation_given_event_and_probability():
    space = FiniteProbabilitySpace([0, 1, 2, 3], [0.25] * 4)
    x = space.random_variable([1.0, 2.0, 3.0, 4.0])
    assert conditional_expectation_given_event(space, x, {1, 2}) == pytest.approx(2.5)
    assert conditional_probability_given_event(space, {1, 2}, {1, 3}) == pytest.approx(0.5)


def test_independence_and_conditional_characterization():
    space = FiniteProbabilitySpace([0, 1, 2, 3], [0.25] * 4)
    x = space.random_variable([0.0, 0.0, 1.0, 1.0])
    y = space.random_variable([0.0, 1.0, 0.0, 1.0])
    assert independent_random_variables(space, x, y)
    partition = Partition.generated_by(y)
    assert conditional_characterization_error(space, x, partition) <= 1e-12


def test_convex_transform_of_martingale_is_submartingale():
    space = FiniteProbabilitySpace([0, 1, 2, 3], [0.25] * 4)
    eps1 = space.random_variable([1.0, 1.0, -1.0, -1.0])
    eps2 = space.random_variable([1.0, -1.0, 1.0, -1.0])
    x0 = space.random_variable([0.0] * 4)
    x1 = x0 + eps1
    x2 = x1 + eps2
    filtration = Filtration.natural([x0, x1, x2])
    mart = Martingale([x0, x1, x2], filtration)
    abs_process = transform_martingale(mart, abs)
    assert abs_process.is_submartingale()


def test_stopped_martingale_helper():
    space = FiniteProbabilitySpace([0, 1], [0.5, 0.5])
    x0 = space.random_variable([0.0, 0.0])
    x1 = space.random_variable([1.0, -1.0])
    trivial = space.partition([space.outcomes])
    full = space.partition([{0}, {1}])
    filtration = Filtration([trivial, full])
    mart = Martingale([x0, x1], filtration)
    stopping = StoppingTime.from_values(space, {0: 1, 1: 1}, filtration)
    stopped = stopped_martingale(mart, stopping)
    assert stopped.is_martingale()
