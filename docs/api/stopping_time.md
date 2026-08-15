# StoppingTime

`StoppingTime` represents a discrete stopping time with values in $\mathbb N\cup\{\infty\}$ with respect to a filtration. The construction enforces the finite-time stopping-time condition from the course model.

## Constructor

```python
StoppingTime(space, values, filtration)
```

### Parameters

| Parameter | Description |
|---|---|
| `space` | Underlying `FiniteProbabilitySpace`. |
| `values` | Outcome-to-stopping-time mapping. |
| `filtration` | Filtration with respect to which the stopping condition is checked. |

## Class methods

- `from_values(space, values, filtration)` — explicit construction helper from outcome-wise values.

## Properties

| Property | Meaning |
|---|---|
| `space` | Underlying probability space. |
| `values` | Outcome-wise stopping times. |
| `filtration` | Filtration used for the definition. |

## Methods

- `minimum(other)` — construct $T\wedge S$.
- `maximum(other)` — construct $T\vee S$.
- `add(other)` — construct $T+S$ under the finite representation.

The stopping-time validity is enforced during construction.

## Example

```python
T = StoppingTime.from_values(space, values, filtration)
S = T.minimum(other)
print(S.values)
```

## Related course material

[Chapter 5 — Martingales à temps discret](../course_chapitre5.md)
