"""Complete Chapter 4 finite conditional-expectation demonstration."""

from stochx.stochastic import FiniteProbabilitySpace, Partition

space = FiniteProbabilitySpace([1, 2, 3, 4], [0.25] * 4)
X = space.random_variable([1.0, 2.0, 3.0, 4.0], name="X")
Y = space.random_variable([0.0, 0.0, 1.0, 1.0], name="Y")
G = Partition.from_blocks([{1, 2}, {3, 4}], space)

print("P({1,2}) =", space.probability({1, 2}))
print("X =", X.array())
print("E[X] =", X.expectation())
print("E[X | {1,2}] =", space.conditional_expectation_given_event(X, {1, 2}))
print("E[X | G] =", space.conditional_expectation(X, G).array())
print("E[X | Y] =", space.conditional_expectation_given(X, Y).array())
print("P({1,2} | G) =", space.conditional_probability({1, 2}, G).array())
print("total expectation =", space.total_expectation(X, G))
print("variance =", space.variance(X))
print("conditional variance =", space.conditional_variance(X, G).array())
print("total variance =", space.total_variance(X, G))
print("measurable on G =", space.check_measurable(Y, G))
print("event probability =", space.conditional_probability_given_event({1}, {1, 2}))
print("event conditional expectation =", space.conditional_expectation_given_event(X, {1, 2}))
print("X independent of Y =", space.are_independent(X, Y))
