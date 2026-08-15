import pytest

from optiflowx.stochastic import FiniteProbabilitySpace, Partition


def test_conditional_event_helpers_and_independence():
    space = FiniteProbabilitySpace([0, 1, 2, 3], [0.25] * 4)
    x = space.random_variable([1.0, 2.0, 3.0, 4.0])
    assert space.conditional_expectation_given_event(x, {1, 2}) == pytest.approx(2.5)
    assert space.conditional_probability_given_event({1, 2}, {1, 3}) == pytest.approx(0.5)
    a = space.random_variable([0.0, 0.0, 1.0, 1.0])
    b = space.random_variable([0.0, 1.0, 0.0, 1.0])
    assert space.are_independent(a, b)
    G = Partition.generated_by(b)
    assert space.conditional_characterization_error(a, G) <= 1e-12
