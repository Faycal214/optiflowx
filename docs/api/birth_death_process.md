# BirthDeathProcess

`BirthDeathProcess` is the birth-death specialization of a continuous-time Markov chain. From state $k$, transitions occur only to $k+1$ and $k-1$ with rates $\lambda_k$ and $\mu_k$.

## Constructor

```python
BirthDeathProcess(birth_rates, death_rates, *, max_state=None)
```

### Parameters

| Parameter | Description |
|---|---|
| `birth_rates` | Sequence or callable specification for $\lambda_k$. |
| `death_rates` | Sequence or callable specification for $\mu_k$. |
| `max_state` | Optional upper state bound required for finite matrix construction. |

## Properties

| Property | Meaning |
|---|---|
| `max_state` | Optional finite state-space bound. |
| `generator` | Generator matrix $Q$ for a finite bounded model. |
| `jump_chain_matrix` | Embedded jump-chain matrix for a finite bounded model. |

## Class methods

- `finite(birth_rates, death_rates)` — construct a finite process from equally sized rate sequences.
- `linear(birth_rate, death_rate, immigration=0.0, emigration=0.0, max_state=None)` — construct linear rates.
- `pure_immigration(rate)` — construct the pure-immigration case.
- `pure_birth(rate)` — construct the pure-birth case.
- `pure_death(rate)` — construct the pure-death case.

## Methods

- `birth_rate(k)` — return $\lambda_k$.
- `death_rate(k)` — return $\mu_k$.
- `generator_matrix()` — build the finite generator $Q$.
- `to_ctmc()` — convert a finite birth-death model into `ContinuousTimeMarkovChain`.
- `jump_chain_matrix()` — construct the embedded DTMC matrix.
- `jump_chain()` — return the embedded `MarkovChain`.
- `kolmogorov_derivative(probabilities)` — evaluate the birth-death form of $p'(t)=p(t)Q$.
- `stationary_weights(n_terms)` — compute the finite product weights used by the stationary law.
- `stationary_weights_at(n_terms)` — canonical alias for the stationary product weights.
- `stationary_distribution(n_terms=None)` — normalize the stationary weights or derive the finite stationary law.
- `pure_birth_probability(n, t, rate=...)` — pure-birth probability formula.
- `pure_death_probability(n, t, initial_population=..., rate=...)` — pure-death probability formula.
- `pure_birth_reciprocal_rate_sum(n_terms)` — partial reciprocal-rate sum for the pure-birth model.
- `simulate(...)` — simulate the process where supported by the implementation.

## Example

```python
from optiflowx.stochastic import BirthDeathProcess

process = BirthDeathProcess.linear(
    birth_rate=0.8,
    death_rate=0.6,
    max_state=10,
)

print(process.generator)
print(process.to_ctmc().states)
```

## Relationship with CMTC

A birth-death process is represented through the same $Q$-matrix framework as a finite CMTC. The specialized class keeps birth/death rates explicit and exposes conversion to the general CTMC object.

## Related course material

[Chapter 3 — CMTC](../course_chapitre3.md)
