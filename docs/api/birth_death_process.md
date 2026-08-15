# BirthDeathProcess

`BirthDeathProcess` is the birth-death specialization of a continuous-time Markov chain. From state $k$, transitions occur only to $k+1$ and $k-1$ with rates $\lambda_k$ and $\mu_k$.

## Constructor

```python
BirthDeathProcess(birth_rate, death_rate, max_state=None, tolerance=1e-12)
```

### Properties

| Property | Meaning |
|---|---|
| `generator` | Generator matrix built from the birth/death rates. |
| `generator_matrix` | Alias for `generator`. |
| `jump_chain_matrix` | Embedded jump-chain matrix. |
| `max_state` | Optional finite state-space bound. |

## Methods

- `birth_rate(k)` — return $\lambda_k$.
- `death_rate(k)` — return $\mu_k$.
- `generator()` — construct the birth-death generator when a callable form is used.
- `jump_chain()` — return the embedded DTMC.
- `kolmogorov_derivative(probabilities)` — derivative of the state-probability vector under the birth-death generator.
- `stationary_weights(...)` — compute proportional stationary weights.
- `stationary_distribution(...)` — normalize stationary weights when the finite model admits one.
- `pure_immigration_probability(...)` — probability for the pure-immigration case covered by the implementation.
- `pure_birth_probability(...)` — probability for the pure-birth case.
- `pure_death_probability(...)` — probability for the pure-death case.
- `pure_birth_reciprocal_rate_sum(...)` — reciprocal-rate quantity used for the pure-birth model.
- `simulate(...)` — simulate the birth-death process subject to the supplied horizon/bound.

## Example

```python
from optiflowx.stochastic import BirthDeathProcess

process = BirthDeathProcess(
    birth_rate=lambda k: 2.0,
    death_rate=lambda k: 1.0 if k > 0 else 0.0,
    max_state=10,
)

print(process.generator())
print(process.stationary_distribution())
```

## Relationship with CMTC

A birth-death process is represented through the same $Q$-matrix framework as a finite CMTC. The API keeps the specialized rate representation while allowing conversion to the general continuous-time Markov-chain object where supported.

## Related course material

[Chapter 3 — CMTC](../course_chapitre3.md)
