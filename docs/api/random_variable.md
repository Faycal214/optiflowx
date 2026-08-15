# RandomVariable

`RandomVariable` represents a discrete random variable on a `FiniteProbabilitySpace`.

$$X:\Omega\to\mathbb R.$$

## Constructor

```python
RandomVariable(space, values, name=None)
```

### Properties

| Property | Meaning |
|---|---|
| `space` | Underlying finite probability space. |
| `values` | Outcome-to-value mapping. |
| `support` | Values taken by the random variable. |
| `name` | Optional display name. |

## Methods

- `expected_value()` — compute $\mathbb E[X]$.
- `variance()` — compute $\operatorname{Var}(X)$.
- `covariance(other)` — compute covariance with another random variable.
- `probability(value)` — probability of a value.
- `distribution()` — return the induced finite distribution.
- `transform(function, name=None)` — apply a function pointwise.
- `conditional_expectation(partition)` — conditional expectation with respect to a partition when supported directly by the object.

Arithmetic operations such as `X + Y`, `X - Y`, `X * Y`, and `-X` preserve the public `RandomVariable` type.

## Example

```python
space = ...
X = space.random_variable({"H": 1.0, "T": 0.0}, name="X")
Y = X.transform(lambda x: x + 1.0, name="Y")

print(X.expected_value())
print(Y.values)
```

## Related course material

[Chapter 4 — Espérance conditionnelle](../course_chapitre4.md)
