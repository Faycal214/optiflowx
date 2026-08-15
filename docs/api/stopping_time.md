# StoppingTime

`StoppingTime` represents a discrete stopping time with respect to a filtration. The defining condition is that the event

$$\{T\le n\}\in\mathcal F_n$$

holds at every finite time represented by the object.

## Constructor

```python
StoppingTime(values, filtration)
```

## Properties

| Property | Meaning |
|---|---|
| `values` | Time at which stopping occurs on each outcome. |
| `filtration` | Filtration relative to which stopping is tested. |
| `n_steps` | Number of represented time levels. |

## Methods

- `is_stopping_time()` — validate the stopping-time condition.
- `minimum(other)` — minimum of two stopping times.
- `maximum(other)` — maximum of two stopping times.
- `add(other)` — combine stopping-time values under the implemented finite construction.

## Example

```python
T = StoppingTime(values, filtration)
print(T.is_stopping_time())
T_min = T.minimum(S)
```

## Related course material

[Chapter 5 — Martingales à temps discret](../course_chapitre5.md)
