# OptiFlowX

[![CI](https://github.com/Faycal214/optiflowx/actions/workflows/test.yml/badge.svg)](https://github.com/Faycal214/optiflowx/actions/workflows/test.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-1081C2?style=flat)](https://faycal214.github.io/optiflowx/)
[![Python](https://img.shields.io/badge/python-3.10%2B-1081C2?style=flat)](https://www.python.org/)
[![License](https://img.shields.io/github/license/Faycal214/optiflowx?style=flat)](LICENSE)

OptiFlowX is a lightweight Python library for turning stochastic-process mathematics into executable, validated, and testable objects.

It is designed around a simple idea: each mathematical object should have a clear Python representation, a predictable API, numerical validation, and runnable examples.

## What makes OptiFlowX different

OptiFlowX is not limited to discrete-time Markov chains. Its public stochastic API is organized around several connected mathematical objects:

| Area | Main objects |
|---|---|
| Discrete-time Markov chains | `MarkovChain` |
| Poisson processes | `PoissonProcess`, `NonHomogeneousPoissonProcess` |
| Continuous-time Markov chains | `ContinuousTimeMarkovChain`, `CTMCPath` |
| Birth-death processes | `BirthDeathProcess` |
| Finite probability spaces | `FiniteProbabilitySpace`, `RandomVariable`, `Partition` |
| Conditional expectation | `FiniteProbabilitySpace`, `RandomVariable` |
| Filtrations and martingales | `Filtration`, `Martingale`, `StoppingTime`, `StoppedProcess` |

Two features are particularly central to the library:

- **CTMC numerical flexibility:** transition probabilities can be evaluated using the matrix-exponential route or a uniformization implementation.
- **Mathematical continuity:** finite conditional expectation, filtrations, martingales, and stopping times are first-class public objects rather than separate utilities.

## Installation

```bash
python -m pip install optiflowx
```

For development:

```bash
python -m pip install -e ".[dev]"
```

For documentation development:

```bash
python -m pip install -e ".[docs]"
mkdocs serve
```

## Quick start

### Discrete-time Markov chain

```python
import numpy as np
from optiflowx.stochastic import MarkovChain, empirical_state_frequencies

P = [
    [0.7, 0.3],
    [0.4, 0.6],
]

chain = MarkovChain(P, states=["A", "B"])

print(chain.n_step_transition(5))
print(chain.stationary_distribution())

path = chain.simulate(
    10_000,
    initial_state="A",
    rng=np.random.default_rng(0),
)
print(empirical_state_frequencies(path, chain.states))
```

### Continuous-time Markov chain

```python
from optiflowx.stochastic import ContinuousTimeMarkovChain

Q = [
    [-2.0, 2.0],
    [1.0, -1.0],
]

chain = ContinuousTimeMarkovChain(Q, states=["A", "B"])

print(chain.transition_matrix(2.0))
print(chain.transition_matrix_at(2.0, method="uniformization"))
```

## Public API

The public stochastic namespace is available from `optiflowx.stochastic`:

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
    empirical_state_frequencies,
)
```

The complete reference is maintained in the [API documentation](https://faycal214.github.io/optiflowx/).

## Examples

Every major mathematical area has a runnable example, and `examples/07_api_operations.py` provides a broader public-API gallery.

```text
examples/
├── 01_discrete_markov_chain.py
├── 02_poisson_process.py
├── 03_continuous_markov_chain.py
├── 04_birth_death_process.py
├── 05_conditional_expectation.py
├── 06_martingale.py
└── 07_api_operations.py
```

The CI suite executes every `examples/*.py` file.

## Documentation

The documentation site separates three concerns:

- **Course material** for the mathematical development.
- **API Reference** for Python classes, properties, methods, validation rules, and examples.
- **Worked Examples** for end-to-end executable usage.

Start at the [documentation site](https://faycal214.github.io/optiflowx/).

## Development and quality gates

The repository uses GitHub Actions to run the stochastic test suite on Python 3.10, 3.11, and 3.12. The CI pipeline also checks:

- public API docstring coverage;
- API-reference page coverage;
- documentation structure;
- runnable example coverage;
- strict MkDocs builds.

Run the main stochastic suite locally with:

```bash
pytest -q tests/test_stochastic_*.py --disable-warnings
```

Run the release-surface checks with:

```bash
pytest -q \
  tests/test_docstring_coverage.py \
  tests/test_stochastic_example_coverage.py \
  tests/test_api_documentation_coverage.py \
  tests/test_documentation_coverage.py
```

Build the package locally before a release:

```bash
python -m build
python -m twine check dist/*
```

## Versioning

OptiFlowX follows semantic versioning for public API changes:

- `MAJOR` for incompatible public API changes;
- `MINOR` for backwards-compatible features;
- `PATCH` for backwards-compatible fixes.

The package version is defined once in `optiflowx/__init__.py` and is used by the build configuration, avoiding separate version values that can drift.

## Release status

OptiFlowX is currently in the early development stage. PyPI publishing is prepared through a tag-based release workflow, but releases are not automatically published until the repository's PyPI trusted publisher is configured.

## License

MIT
