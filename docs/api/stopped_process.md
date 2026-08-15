# StoppedProcess

`StoppedProcess` represents the discrete process stopped at a stopping time $T$:

$$X_n^T=X_{n\wedge T}.$$

## Constructor

```python
StoppedProcess(process, stopping_time)
```

## Properties

| Property | Meaning |
|---|---|
| `process` | Original process. |
| `stopping_time` | Stopping time used for the construction. |
| `n_steps` | Number of represented time levels. |

## Methods

- `values(n)` — obtain the stopped value sequence through time `n`.
- `sequence()` — return the stopped process sequence.
- `terminal_value()` — return the terminal stopped value.

## Example

```python
M = Martingale(process, filtration)
T = StoppingTime(values, filtration)
MT = M.stopped(T)

print(MT.sequence())
print(MT.terminal_value())
```

## Related course material

[Chapter 5 — Martingales à temps discret](../course_chapitre5.md)
