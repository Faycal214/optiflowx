# OptiFlowX

OptiFlowX is a Python library for working with the stochastic-process material developed in the USTHB MSPRO **Processus Aléatoires** course. The implementation follows the five supplied PDFs and keeps the mathematical objects executable, testable, and transparent.

## Course coverage

| Chapter | Course material | Main API |
|---|---|---|
| 1 | Discrete-time Markov chains (CMTD) | `MarkovChain` |
| 2 | Poisson processes | `PoissonProcess`, `NonHomogeneousPoissonProcess` |
| 3 | Continuous-time Markov chains (CMTC) | `ContinuousTimeMarkovChain` |
| 3 | Birth-death processes | `BirthDeathProcess` |
| 4 | Conditional expectation | `FiniteProbabilitySpace`, `RandomVariable`, `Partition` |
| 5 | Discrete-time martingales | `Filtration`, `Martingale`, `StoppingTime`, `StoppedProcess` |

The source boundary is strict: features are added only when they correspond to definitions, propositions, formulas, examples, or constructions present in the five PDFs.

## Mathematical documentation

The repository contains a mathematical reference explaining the definitions and formulas behind every public component:

- [CMTD](docs/cmt_d.md)
- [Processus de Poisson](docs/poisson.md)
- [CMTC](docs/cmtc.md)
- [Processus de naissance et de mort](docs/birth_death.md)
- [Espérance conditionnelle](docs/conditional_expectation.md)
- [Martingales à temps discret](docs/martingales.md)
- [Complete API map](docs/api.md)

The site can be built with MkDocs:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Worked examples

Each chapter has a runnable example under [`examples/`](examples/):

```text
examples/
├── 01_discrete_markov_chain.py
├── 02_poisson_process.py
├── 03_continuous_markov_chain.py
├── 04_birth_death_process.py
├── 05_conditional_expectation.py
└── 06_martingale.py
```

## Installation

```bash
python -m pip install optiflowx
```

For development:

```bash
python -m pip install -e .
```

## Quick example

```python
import numpy as np
from optiflowx.stochastic import MarkovChain, empirical_state_frequencies

P = [[0.7, 0.3],
     [0.4, 0.6]]

chain = MarkovChain(P, states=["A", "B"])

print(chain.n_step_transition(5))
print(chain.stationary_distribution())

path = chain.simulate(10_000, initial_state="A", rng=np.random.default_rng(0))
print(empirical_state_frequencies(path, chain.states))
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

Additional theory helpers are exported for the mathematical results that sit above the core objects, including first-return probabilities, mean return times, CTMC stationary laws from the jump chain, conditional expectation by events, independence checks, occupation fractions, and martingale transformations.

## Testing

```bash
pytest -q tests/test_stochastic_*.py --disable-warnings
```

GitHub Actions runs the stochastic suite on Python 3.10, 3.11, and 3.12.

## Source basis

The implementation is based strictly on the five supplied USTHB MSPRO 2024–2025 course PDFs:

1. CMTD / discrete-time Markov chains.
2. Poisson processes.
3. CMTC / continuous-time Markov chains and birth-death processes.
4. Conditional expectation.
5. Generalities on discrete-time martingales.

## License

MIT
