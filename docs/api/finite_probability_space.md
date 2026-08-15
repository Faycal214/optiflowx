# FiniteProbabilitySpace

`FiniteProbabilitySpace` represents a finite sample space with an explicit probability mass function and provides the finite conditional-expectation operations exposed by StochX.

## Constructor

```python
FiniteProbabilitySpace(outcomes, probabilities)
```

### Parameters

| Parameter | Description |
|---|---|
| `outcomes` | Finite set or ordered collection of outcomes. |
| `probabilities` | Probability assigned to each outcome. |

## Properties

| Property | Meaning |
|---|---|
| `outcomes` | Ordered outcomes of the sample space. |
| `probabilities` | Validated probability masses. |
| `n_outcomes` | Number of outcomes. |

## Methods

- `probability(event)` / `probability_of(event)` — compute the probability of an event.
- `random_variable(values, name=None)` — create a `RandomVariable` on the space.
- `partition(blocks)` — create a `Partition`.
- `conditional_probability_given_event(event, condition)` — compute conditional probability given an event.
- `conditional_expectation_given_event(random_variable, event)` — compute the scalar conditional expectation given an event.
- `conditional_expectation(random_variable, partition)` — compute \(E[X\mid\mathcal G]\) on a partition.
- `conditional_expectation_given(random_variable, condition)` — condition on a supported finite random variable or conditioning object.
- `conditional_probability(event, condition)` — compute conditional probability on a finite conditioning object.
- `are_partitions_independent(first, second)` — test partition independence.
- `are_independent(first, second)` — test supported independence relations.
- `conditional_characterization_error(random_variable, partition)` — evaluate the finite characterization error.
- `total_expectation(random_variable, partition)` — apply the law of total expectation.
- `tower(random_variable, fine, coarse)` — apply the tower property.
- `pull_out(multiplier, random_variable, partition)` — apply the pull-out identity.
- `conditional_variance(random_variable, partition)` — compute conditional variance.
- `conditional_covariance(first, second, partition)` — compute conditional covariance.
- `total_variance(random_variable, partition)` — apply the law of total variance.
- `total_covariance(first, second, partition)` — apply the law of total covariance.
- `l2_projection(random_variable, partition)` — compute the finite \(L^2\) projection.

## Example

```python
from stochx.stochastic import FiniteProbabilitySpace

space = FiniteProbabilitySpace(
    outcomes=["H", "T"],
    probabilities=[0.6, 0.4],
)
X = space.random_variable({"H": 1.0, "T": 0.0}, name="X")
print(space.probability_of({"H"}))
print(X.expected_value())
print(space.total_expectation(X, space.partition([{ "H" }, { "T" } ])))
```

## Related API

[`RandomVariable`](random_variable.md) represents variables defined on the space.
