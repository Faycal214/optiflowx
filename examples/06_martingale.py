"""Chapter 5: discrete-time martingale example."""

from stochx.stochastic import Filtration, FiniteProbabilitySpace, Martingale

space = FiniteProbabilitySpace([0, 1, 2, 3], [0.25] * 4)
eps1 = space.random_variable([1.0, 1.0, -1.0, -1.0], name="epsilon_1")
eps2 = space.random_variable([1.0, -1.0, 1.0, -1.0], name="epsilon_2")
x0 = space.random_variable([0.0] * 4, name="X_0")
x1 = x0 + eps1
x2 = x1 + eps2

filtration = Filtration.natural([x0, x1, x2])
mart = Martingale([x0, x1, x2], filtration)

print("is martingale:", mart.is_martingale())
print("conditional future E[X_2|F_0]:", mart.conditional_future(0, 2).array())
print("expectations:", mart.expectations())

absolute = mart.transform(abs)
print("|X_n| is a submartingale:", absolute.is_submartingale())
