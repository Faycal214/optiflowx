# OptiFlowX

OptiFlowX is a Python library for working with stochastic-process mathematics through executable, testable objects.

## Coverage

| Chapter | Mathematical area | Main API |
|---|---|---|
| 1 | Discrete-time Markov chains (DTMC) | `MarkovChain` |
| 2 | Poisson processes | `PoissonProcess`, `NonHomogeneousPoissonProcess` |
| 3 | Continuous-time Markov chains (CTMC) | `ContinuousTimeMarkovChain` |
| 3 | Birth-death processes | `BirthDeathProcess` |
| 4 | Conditional expectation | `FiniteProbabilitySpace`, `RandomVariable`, `Partition` |
| 5 | Discrete-time martingales | `Filtration`, `Martingale`, `StoppingTime`, `StoppedProcess` |

## Mathematical documentation

The repository contains a mathematical reference explaining the definitions, properties, and formulas behind the public components:

- [DTMC](docs/cmt_d.md)
- [Poisson processes](docs/poisson.md)
- [CTMC](docs/cmtc.md)
- [Birth-death processes](docs/birth_death.md)
- [Conditional expectation](docs/conditional_expectation.md)
- [Discrete-time martingales](docs/martingales.md)
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

## Testing

```bash
pytest -q tests/test_stochastic_*.py --disable-warnings
```

GitHub Actions runs the stochastic suite on Python 3.10, 3.11, and 3.12.

## License

MIT
