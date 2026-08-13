# OptiFlowX

OptiFlowX is a Python library for working with the stochastic-process material developed in the USTHB MSPRO **Processus Aléatoires** course. The stochastic API is organized around the five course chapters and is intended to make the definitions, formulas, and constructions of the PDFs executable and testable.

## Course coverage

| Chapter | Course material | Main API |
|---|---|---|
| 1 | Discrete-time Markov chains (CMTD) | `MarkovChain` |
| 2 | Poisson processes | `PoissonProcess`, `NonHomogeneousPoissonProcess` |
| 3 | Continuous-time Markov chains and birth-death processes | `ContinuousTimeMarkovChain`, `BirthDeathProcess` |
| 4 | Conditional expectation | `FiniteProbabilitySpace`, `RandomVariable`, `Partition` |
| 5 | Discrete-time martingales | `Filtration`, `Martingale`, `StoppingTime`, `StoppedProcess` |

The implementation follows the finite/discrete computational setting used throughout the supplied course material. Features are added only when they correspond to definitions, propositions, formulas, examples, or constructions present in the PDFs.

## Installation

```bash
python -m pip install optiflowx
```

For development:

```bash
python -m pip install -e . --no-deps
python -m pip install numpy scipy pytest
```

## Quick examples

### Discrete-time Markov chain

```python
import numpy as np
from optiflowx.stochastic import MarkovChain

P = [[0.7, 0.3],
     [0.4, 0.6]]

chain = MarkovChain(P, states=["A", "B"])

print(chain.n_step_transition(5))
print(chain.stationary_distribution())
print(chain.simulate(10, initial_state="A", rng=np.random.default_rng(0)))
```

### Poisson process

```python
from optiflowx.stochastic import PoissonProcess

process = PoissonProcess(rate=2.0)
print(process.count_probability(n=4, t=3.0))
print(process.arrival_times(5))
```

### Continuous-time Markov chain

```python
from optiflowx.stochastic import ContinuousTimeMarkovChain

Q = [[-2.0, 2.0],
     [ 1.0, -1.0]]

chain = ContinuousTimeMarkovChain(Q)
print(chain.transition_matrix(0.5))
print(chain.stationary_distribution())
```

### Conditional expectation on a finite probability space

```python
from optiflowx.stochastic import FiniteProbabilitySpace

space = FiniteProbabilitySpace(
    outcomes=[0, 1, 2, 3],
    probabilities=[0.25, 0.25, 0.25, 0.25],
)

X = space.random_variable([1.0, 3.0, 5.0, 7.0])
Y = space.random_variable([0.0, 0.0, 1.0, 1.0])

print(space.conditional_expectation_given(X, Y).array())
```

### Martingale

```python
from optiflowx.stochastic import Filtration, Martingale

filtration = Filtration.natural([X])
martingale = Martingale([X], filtration)
print(martingale.is_martingale())
```

## Public stochastic API

```python
from optiflowx.stochastic import (
    BirthDeathProcess,
    CTMCPath,
    ContinuousTimeMarkovChain,
    FiniteProbabilitySpace,
    Filtration,
    MarkovChain,
    Martingale,
    NonHomogeneousPoissonProcess,
    Partition,
    PoissonProcess,
    RandomVariable,
    StoppedProcess,
    StoppingTime,
)
```

## Design principle

The library separates the computational objects of the course rather than introducing a general-purpose symbolic probability framework. Finite probability spaces and partitions are used for the Chapter 4 material; finite-state transition matrices and generators are used for the Markov-chain chapters; discrete filtrations and stopping times are used for Chapter 5.

## Testing

The branch is tested on Python 3.10, 3.11, and 3.12 with the stochastic test suite:

```bash
pytest -q tests/test_stochastic_*.py --disable-warnings
```

## Source basis

The implementation is based on the five supplied USTHB MSPRO 2024–2025 course PDFs:

1. CMTD / discrete-time Markov chains.
2. Poisson processes.
3. CMTC / continuous-time Markov chains and birth-death processes.
4. Conditional expectation.
5. Generalities on discrete-time martingales.

## License

MIT
