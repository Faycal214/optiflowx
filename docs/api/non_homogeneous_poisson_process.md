# NonHomogeneousPoissonProcess

`NonHomogeneousPoissonProcess` represents a Poisson process with time-dependent intensity
\(\lambda(t)\) and mean function

$$m(t)=\int_0^t \lambda(x)\,dx.$$ 

## Constructor

```python
NonHomogeneousPoissonProcess(intensity_function, mean_function=None)
```

### Parameters

| Parameter | Description |
|---|---|
| `intensity_function` | Callable intensity \(\lambda(t)\). |
| `mean_function` | Optional callable mean function \(m(t)\). |

## Properties

| Property | Meaning |
|---|---|
| `intensity_function` | The intensity callable. |
| `mean_function` | The mean-function callable. |

## Methods

- `mean(t)` — evaluate \(m(t)\).
- `count_probability(n, t)` — compute the count probability at time `t`.
- `increment_probability(n, s, t)` — compute the increment probability on `[s, t]`.
- `simulate(t_max, rng=None)` — generate events up to a finite horizon.

## Example

```python
from stochx.stochastic import NonHomogeneousPoissonProcess

process = NonHomogeneousPoissonProcess(
    intensity=lambda t: 1.0 + t,
    mean_function=lambda t: t + 0.5 * t**2,
)

print(process.intensity_function(2.0))
print(process.mean(2.0))
print(process.count_probability(2, 2.0))
```

## Related API

[`PoissonProcess`](poisson_process.md) provides the homogeneous constant-rate case.
