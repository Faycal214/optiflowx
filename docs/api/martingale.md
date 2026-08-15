# Martingale

`Martingale` represents a discrete-time adapted process whose conditional future expectation equals its present value under the supplied filtration.

For a martingale $(M_n)$,

$$\mathbb E[M_{n+1}\mid\mathcal F_n]=M_n.$$

The same object supports the submartingale and supermartingale tests used by the package.

## Constructor

```python
Martingale(process, filtration)
```

## Properties

| Property | Meaning |
|---|---|
| `process` | Underlying discrete-time process. |
| `filtration` | Filtration with respect to which the process is analyzed. |
| `n_steps` | Number of available time steps. |

## Methods

- `value_at(n)` — value/process state at time `n`.
- `conditional_next(n)` — conditional next-step expectation.
- `martingale_residual(n)` — difference defining the martingale condition.
- `is_martingale()` — test the martingale property.
- `is_submartingale()` — test the submartingale property.
- `is_supermartingale()` — test the supermartingale property.
- `conditional_future(n, k)` — conditional future value at a later step.
- `expectations()` — expectations across the time grid.
- `transform(function)` — transform the process pointwise while preserving the appropriate public process representation.
- `doob(random_variable, filtration)` — construct the Doob martingale supported by the implementation.
- `stopped(stopping_time)` — construct the stopped process.

## Example

```python
M = Martingale(process, filtration)
print(M.is_martingale())
print(M.martingale_residual(2))
```

## Related course material

[Chapter 5 — Martingales à temps discret](../course_chapitre5.md)
