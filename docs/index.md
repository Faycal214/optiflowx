# OptiFlowX

**A Python library for stochastic processes, built from the MSPRO stochastic-process course.**

OptiFlowX provides executable mathematical objects for the stochastic-process topics covered by the five supplied USTHB MSPRO *Processus Aléatoires* course PDFs.

The library currently covers:

- discrete-time Markov chains (CMTD);
- continuous-time Markov chains (CMTC);
- Poisson processes;
- birth-death processes;
- finite probability spaces, partitions, random variables, and conditional expectation;
- filtrations, martingales, stopping times, and stopped processes.

## Installation

```bash
pip install optiflowx
```

## Quick example

Create a discrete-time Markov chain from its transition matrix:

```python
import numpy as np
from optiflowx.stochastic import MarkovChain

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

The five course chapters present the mathematical definitions, notation, results, and hypotheses from the supplied USTHB MSPRO PDFs. The course material is kept separate from the software reference.

### Package / API

The API Reference documents the public OptiFlowX classes, functions, properties, methods, validation rules, and numerical behavior.

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

## Source

The mathematical documentation is based on the five supplied USTHB MSPRO *Processus Aléatoires* PDFs. The API documentation describes the corresponding OptiFlowX implementation separately.

**OptiFlowX** — stochastic-process mathematics implemented as a Python library.