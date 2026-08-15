# FiniteProbabilitySpace

`FiniteProbabilitySpace` represents a finite sample space with an explicit probability mass function and provides the main finite conditional-expectation operations.

## Constructor

```python
FiniteProbabilitySpace(outcomes, probabilities)
```

### Parameters

| Parameter | Description |
|---|---|
| `outcomes` | Finite set/ordered collection of outcomes. |
| `probabilities` | Probability assigned to each outcome. |

## Properties

| Property | Meaning |
|---|---|
| `outcomes` | Ordered outcomes of the sample space. |
| `probabilities` | Validated probability masses. |
| `n_outcomes` | Number of outcomes. |

## Methods

- `probability(event)` / `probability_of(event)` — probability of an event.
- `random_variable(values, name=None)` — create a `RandomVariable` on the space.
- `partition(blocks)` — create a validated `Partition`.
- `conditional_probability_given_event(event, condition)` — conditional probability given an event.
- `conditional_expectation_given_event(random_variable, event)` — conditional expectation given an event.
- `conditional_expectation(random_variable, partition)` — conditional expectation with respect to a partition.
- `conditional_expectation_given(random_variable, condition)` — conditional expectation under a supported finite conditioning object.
- `conditional_probability(event, condition)` — probability conditioned on a finite conditioning object.
- `are_partitions_independent(first, second)` — test independence of two partitions.
- `are_independent(first, second)` — test independence of supported random variables/partitions.
- `conditional_characterization_error(random_variable, partition)` — finite conditional-expectation characterization error.
- `total_expectation(random_variable, partition)` — law of total expectation.
- `tower(random_variable, fine, coarse)` — tower property.
- `pull_out(multiplier, random_variable, partition)` — pull-out property under the implemented finite conditions.
- `conditional_variance(random_variable, partition)` — conditional variance.
- `conditional_covariance(first, second, partition)` — conditional covariance.
- `total_variance(random_variable, partition)` — law of total variance.
- `total_covariance(first, second, partition)` — law of total covariance.
- `l2_projection(random_variable, partition)` — finite $L^2$ projection on the conditioning partition.

## Example

```python
from optiflowx.stochastic import FiniteProbabilitySpace

space = FiniteProbabilitySpace(
    outcomes=["H", "T"],
    probabilities=[0.6, 0.4],
)
X = space.random_variable({"H": 1.0, "T": 0.0}, name="X")
print(space.probability_of({"H"}))
print(X.expected_value())
```

## Related course material

[Chapter 4 — Espérance conditionnelle](../course_chapitre4.md)
