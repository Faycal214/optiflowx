# Probability spaces, random variables and conditional expectation

## Finite probability spaces

`FiniteProbabilitySpace` represents a finite sample space with explicit outcome probabilities.

```python
from stochx.stochastic import FiniteProbabilitySpace

space = FiniteProbabilitySpace(
    outcomes=["H", "T"],
    probabilities=[0.5, 0.5],
)
```

The probability vector is validated at construction time.

## Random variables

A `RandomVariable` maps outcomes to numerical values:

```python
from stochx.stochastic import RandomVariable

X = RandomVariable(space, {"H": 1.0, "T": 0.0})
```

The object can expose expectations and other operations supported by the public API.

For a finite space,

$$\mathbb E[X]=\sum_{\omega}X(\omega)\mathbb P(\omega).$$

## Partitions

A `Partition` groups outcomes into information sets. It provides the structure needed for conditional expectation on a finite probability space.

```python
from stochx.stochastic import Partition

partition = Partition(space, [["H"], ["T"]])
```

## Conditional expectation

For a partition $\mathcal G$, the conditional expectation is constant on each partition cell and preserves conditional averages:

$$\mathbb E[X\mid\mathcal G].$$

The package exposes this concept through the probability-space/random-variable/partition API rather than requiring manual construction of indicator vectors.

## Why explicit objects?

These classes make probability-theory assumptions executable. Invalid probabilities, incompatible outcome labels or malformed partitions should fail at the API boundary rather than quietly producing a meaningless numerical result.

## API reference

See the pages for `FiniteProbabilitySpace`, `RandomVariable` and `Partition` for the complete public surface.
