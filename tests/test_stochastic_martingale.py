import numpy as np
import pytest

from stochx.stochastic import Filtration, FiniteProbabilitySpace, Martingale, StoppingTime


def fair_coin_space():
    return FiniteProbabilitySpace([0, 1], [0.5, 0.5])


def test_natural_filtration_and_fair_random_walk_are_martingale():
    space = FiniteProbabilitySpace([0, 1, 2, 3], [0.25] * 4)
    eps1 = space.random_variable([1.0, 1.0, -1.0, -1.0])
    eps2 = space.random_variable([1.0, -1.0, 1.0, -1.0])
    x0 = space.random_variable([0.0] * 4)
    x1 = x0 + eps1
    x2 = x1 + eps2
    filtration = Filtration.natural([x0, x1, x2])
    mart = Martingale([x0, x1, x2], filtration)

    assert mart.is_martingale()
    np.testing.assert_allclose(mart.expectations(), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(mart.conditional_future(0, 2).array(), x0.array())


def test_stopping_time_and_stopped_process():
    space = fair_coin_space()
    x0 = space.random_variable([0.0, 0.0])
    x1 = space.random_variable([1.0, -1.0])
    full = space.partition([{0}, {1}])
    filtration = Filtration([full, full])
    stopping = StoppingTime.from_values(space, {0: 1, 1: 0}, filtration)
    stopped = Martingale([x0, x1], filtration).stopped(stopping)
    np.testing.assert_allclose(stopped.values(1).array(), [1.0, 0.0])
    terminal = stopped.terminal_value()
    np.testing.assert_allclose(terminal.array(), [1.0, 0.0])


def test_invalid_stopping_time_is_rejected():
    space = fair_coin_space()
    x0 = space.random_variable([0.0, 0.0])
    x1 = space.random_variable([1.0, -1.0])
    filtration = Filtration.natural([x0, x1])
    with pytest.raises(ValueError):
        StoppingTime.from_values(space, {0: 0, 1: 1}, filtration)


def test_stopping_time_closure():
    space = fair_coin_space()
    trivial = space.partition([space.outcomes])
    full = space.partition([{0}, {1}])
    filtration = Filtration([trivial, full])
    t1 = StoppingTime.from_values(space, {0: 0, 1: 0}, filtration)
    t2 = StoppingTime.from_values(space, {0: 1, 1: 1}, filtration)
    assert t1.minimum(t2).values == {0: 0, 1: 0}
