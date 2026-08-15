# StochX

**A Python library for stochastic processes and applied probability.**

The library currently covers:

- discrete-time Markov chains (DTMC);
- continuous-time Markov chains (CTMC);
- Poisson processes;
- birth-death processes;
- finite probability spaces, partitions, random variables, and conditional expectation;
- filtrations, martingales, stopping times, and stopped processes.

## Installation

```bash
pip install stochx
```

## Quick example

Create a discrete-time Markov chain from its transition matrix:

```python
import numpy as np
from stochx.stochastic import MarkovChain

P = np.array([
    [0.8, 0.2],
    [0.3, 0.7],
])

chain = MarkovChain(P, states=["A", "B"])

print(chain.states)
print(chain.transition_matrix)
print(chain.stationary_distribution())
```

## Documentation

The documentation is divided into three parts.

### Course material

The course chapters present the mathematical definitions, notation, results, and hypotheses used throughout the stochastic-process material. The mathematics is kept separate from the software reference.

### Package / API

The API Reference documents the public StochX classes, functions, properties, methods, validation rules, and numerical behavior.

### Examples

Worked examples show how the mathematical objects are represented and used in Python.

## Supported stochastic models

| Model | Main class |
|---|---|
| Discrete-time Markov chain | `MarkovChain` |
| Continuous-time Markov chain | `ContinuousTimeMarkovChain` |
| Poisson process | `PoissonProcess` |
| Birth-death process | `BirthDeathProcess` |
| Finite probability space | `FiniteProbabilitySpace` |
| Random variable | `RandomVariable` |
| Partition | `Partition` |
| Filtration | `Filtration` |
| Martingale | `Martingale` |
| Stopping time | `StoppingTime` |
| Stopped process | `StoppedProcess` |

**StochX** — stochastic-process mathematics implemented as a Python library.
