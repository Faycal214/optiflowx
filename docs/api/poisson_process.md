# Poisson processes

## PoissonProcess

`PoissonProcess` represents a homogeneous Poisson process with constant rate $\lambda>0$.

### Constructor

```python
PoissonProcess(rate)
```

### Properties

| Property | Meaning |
|---|---|
| `rate` | Constant intensity $\lambda$. |
| `lambda_` | Python-safe alias for `rate`. |

### Methods

- `count_probability(n, t)` — compute $\mathbb P(N(t)=n)$.
- `increment_probability(n, s, t)` — compute the increment law.
- `interarrival_samples(n, rng=None)` — sample exponential inter-arrival times.
- `arrival_times(n, rng=None)` — sample cumulative arrival times.
- `conditional_first_arrival_cdf(y, s)` — conditional first-arrival distribution.
- `conditional_arrival_times(k, s)` — ordered arrival times conditioned on the count.
- `superpose(other)` — superpose two homogeneous Poisson processes.
- `split(probability)` — split the process by Bernoulli thinning.
- `simulate(t_max, rng=None)` — generate event times up to a horizon.

### Example

```python
from stochx.stochastic import PoissonProcess

process = PoissonProcess(rate=3.0)
print(process.count_probability(2, 1.0))
print(process.arrival_times(5))
```

## NonHomogeneousPoissonProcess

`NonHomogeneousPoissonProcess` represents a Poisson process with intensity $\lambda(t)$ and mean function

$$m(t)=\int_0^t\lambda(x)\,dx.$$

### Constructor

```python
NonHomogeneousPoissonProcess(intensity_function, mean_function=None)
```

### Properties

| Property | Meaning |
|---|---|
| `intensity_function` | Callable intensity $\lambda(t)$. |
| `mean_function` | Callable mean function $m(t)$ when supplied or constructed. |

### Methods

- `mean(t)` — compute $m(t)$.
- `count_probability(n, t)` — count probability using the mean function.
- `increment_probability(n, s, t)` — increment law on $[s,t]$.
- `simulate(t_max, rng=None)` — simulate event times under the intensity.

## Related course material

[Chapter 2 — Processus de Poisson](../course_chapitre2.md)
