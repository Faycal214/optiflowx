"""Chapter 4: conditional expectation on a finite probability space."""

from optiflowx.stochastic import FiniteProbabilitySpace, Partition

space = FiniteProbabilitySpace(
    outcomes=[0, 1, 2, 3],
    probabilities=[0.25, 0.25, 0.25, 0.25],
)

X = space.random_variable([1.0, 2.0, 3.0, 4.0], name="X")
Y = space.random_variable([0.0, 0.0, 1.0, 1.0], name="Y")

print("E[X]:", X.expectation())
print("E[X|Y]:", space.conditional_expectation_given(X, Y).array())

G = Partition.generated_by(Y)
print("E[X|G]:", space.conditional_expectation(X, G).array())
print("E[X|{1,2}]:", space.conditional_expectation_given_event(X, {1, 2}))
print("P({1,2}|{1,3}):", space.conditional_probability_given_event({1, 2}, {1, 3}))
print("total variance:", space.total_variance(X, G))
print("total covariance:", space.total_covariance(X, Y, G))
print("X independent of Y:", space.are_independent(X, Y))
