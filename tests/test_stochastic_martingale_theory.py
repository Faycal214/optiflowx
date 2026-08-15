from stochx.stochastic import Filtration, FiniteProbabilitySpace, Martingale


def test_martingale_transform():
    space = FiniteProbabilitySpace([0, 1, 2, 3], [0.25] * 4)
    e1 = space.random_variable([1.0, 1.0, -1.0, -1.0])
    e2 = space.random_variable([1.0, -1.0, 1.0, -1.0])
    x0 = space.random_variable([0.0] * 4)
    x1 = x0 + e1
    x2 = x1 + e2
    filtration = Filtration.natural([x0, x1, x2])
    mart = Martingale([x0, x1, x2], filtration)
    assert mart.transform(abs).is_submartingale()
