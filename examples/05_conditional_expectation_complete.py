"""Complete Chapter 4 finite conditional-expectation demonstration."""

from optiflowx.stochastic import FiniteProbabilitySpace, Partition
from optiflowx.stochastic.theory import (
    conditional_expectation_given_event,
    conditional_probability_given_event,
    independent_random_variables,
)

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
print("event probability =", conditional_probability_given_event(space, {1}, {1, 2}))
print("event conditional expectation =", conditional_expectation_given_event(space, X, {1, 2}))
print("X independent of Y =", independent_random_variables(space, X, Y))
