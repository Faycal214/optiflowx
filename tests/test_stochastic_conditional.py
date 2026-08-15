import numpy as np

from stochx.stochastic import FiniteProbabilitySpace, Partition


def make_space():
    return FiniteProbabilitySpace([0, 1, 2, 3], [0.25, 0.25, 0.25, 0.25])


def test_conditioning_on_event():
    space = make_space()
    x = space.random_variable([1.0, 3.0, 5.0, 7.0], name="X")
    assert space.conditional_probability_given_event({0, 1}, {1, 2}) == 0.5
    conditional_value = space.conditional_expectation_given_event(x, {1, 2})
    assert isinstance(conditional_value, float)
    assert conditional_value == 4.0


def test_conditioning_on_full_space_returns_scalar_expectation():
    space = make_space()
    x = space.random_variable([1.0, 3.0, 5.0, 7.0], name="X")

    conditional_value = space.conditional_expectation_given_event(x, {0, 1, 2, 3})

    assert isinstance(conditional_value, float)
    assert conditional_value == x.expectation() == 4.0


def test_conditional_expectation_given_discrete_variable():
    space = make_space()
    x = space.random_variable([1.0, 3.0, 5.0, 7.0], name="X")
    y = space.random_variable([0.0, 0.0, 1.0, 1.0], name="Y")

    cond = space.conditional_expectation_given(x, y)
    np.testing.assert_allclose(cond.array(), [2.0, 2.0, 6.0, 6.0])
    assert cond.expectation() == 4.0
    assert x.expectation() == 4.0


def test_total_expectation_and_tower_property():
    space = make_space()
    x = space.random_variable([1.0, 2.0, 3.0, 4.0])
    coarse = space.partition([{0, 1}, {2, 3}])
    fine = space.partition([{0}, {1}, {2}, {3}])

    cond = space.conditional_expectation(x, coarse)
    np.testing.assert_allclose(cond.array(), [1.5, 1.5, 3.5, 3.5])
    tower = space.tower(x, fine, coarse)
    np.testing.assert_allclose(tower.array(), cond.array())
    assert space.total_expectation(x, coarse) == x.expectation()


def test_pull_out_and_conditional_covariance():
    space = make_space()
    x = space.random_variable([1.0, 2.0, 3.0, 4.0])
    y = space.random_variable([5.0, 5.0, 9.0, 9.0])
    partition = space.partition([{0, 1}, {2, 3}])
    residual = space.pull_out(y, x, partition)
    np.testing.assert_allclose(residual.array(), 0.0)
    covariance = space.conditional_covariance(x, y, partition)
    # Conditional covariance is a random variable on the original
    # finite sample space, hence one value per outcome.
    np.testing.assert_allclose(covariance.array(), [0.0, 0.0, 0.0, 0.0])


def test_indicator_conditional_probability_and_conditional_variance():
    space = make_space()
    x = space.random_variable([0.0, 1.0, 2.0, 3.0])
    partition = space.partition([{0, 1}, {2, 3}])
    probability = space.conditional_probability({1, 3}, partition)
    np.testing.assert_allclose(probability.array(), [0.5, 0.5, 0.5, 0.5])
    variance = space.conditional_variance(x, partition)
    np.testing.assert_allclose(variance.array(), [0.25, 0.25, 0.25, 0.25])


def test_partition_generated_by_variable():
    space = make_space()
    y = space.random_variable([0.0, 0.0, 1.0, 1.0])
    partition = Partition.generated_by(y)
    assert len(partition.blocks) == 2
