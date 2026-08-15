# Filtration

`Filtration` represents an increasing sequence of finite partitions

$$\mathcal F_0\subseteq\mathcal F_1\subseteq\cdots,$$

used to encode the information available to a discrete-time process.

## Constructor

```python
Filtration(partitions)
```

## Properties

| Property | Meaning |
|---|---|
| `partitions` | Ordered finite partitions. |
| `n_steps` | Number of filtration levels. |

## Methods

- `at(n)` — return $\mathcal F_n$.
- `is_adapted(process)` — check adaptation of a process to the filtration.
- `natural(process)` — construct the natural filtration supported by the implementation.

## Example

```python
filtration = Filtration([F0, F1, F2])
print(filtration.at(1))
print(filtration.n_steps)
```

## Related course material

[Chapter 5 — Martingales à temps discret](../course_chapitre5.md)
