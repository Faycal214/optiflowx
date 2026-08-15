# RandomVariable

`RandomVariable` represents a finite-valued random variable on a `FiniteProbabilitySpace`.

$$X:\Omega\to\mathbb R.$$

## Constructor

```python
RandomVariable(space, values, name=None)
```

### Parameters

| Parameter | Description |
|---|---|
| `space` | Underlying finite probability space. |
| `values` | Outcome-to-value mapping. |
| `name` | Optional display name. |

## Properties

| Property | Meaning |
|---|---|
| `space` | Underlying probability space. |
| `values` | Outcome-to-value mapping. |
| `name` | Optional display name. |
| `support` | Distinct values taken by the variable. |

## Methods

- `array()` — return values in probability-space outcome order.
- `expectation()` — compute $\mathbb E[X]$.
- `expected_value()` — canonical alias for `expectation()`.
- `transform(function, name=None)` — construct the pointwise transformation $g(X)$.
- `apply(function, name=None)` — canonical pointwise application API.

Arithmetic operations `+`, `-`, `*`, unary `-`, and their supported reflected forms return another public `RandomVariable` on the same probability space.

## Example

```python
space = ...
X = space.random_variable({"H": 1.0, "T": 0.0}, name="X")
Y = X.transform(lambda x: x + 1.0, name="Y")

print(X.expected_value())
print(Y.support)
```

## Related course material

[Chapter 4 — Espérance conditionnelle](../course_chapitre4.md)
