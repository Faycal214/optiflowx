# StochX

[![PyPI](https://img.shields.io/pypi/v/stochx)](https://pypi.org/project/stochx/)
[![Python](https://img.shields.io/pypi/pyversions/stochx)](https://pypi.org/project/stochx/)
[![CI](https://github.com/Faycal214/stochx/actions/workflows/test.yml/badge.svg)](https://github.com/Faycal214/stochx/actions/workflows/test.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-1081C2?style=flat)](https://faycal214.github.io/stochx/)
[![License](https://img.shields.io/github/license/Faycal214/stochx)](LICENSE)

StochX is a lightweight Python library for turning stochastic-process mathematics, time-series methods, and state-space models into executable, validated, and testable objects.

It is designed around a simple idea: each mathematical object should have a clear Python representation, a predictable API, numerical validation, and runnable examples.

## What makes StochX different

StochX connects course-faithful stochastic-process mathematics with practical time-series analysis and forecasting.

| Area | Main objects |
|---|---|
| Discrete-time Markov chains | `MarkovChain` |
| Poisson processes | `PoissonProcess`, `NonHomogeneousPoissonProcess` |
| Continuous-time Markov chains | `ContinuousTimeMarkovChain`, `CTMCPath` |
| Birth-death processes | `BirthDeathProcess` |
| Finite probability spaces | `FiniteProbabilitySpace`, `RandomVariable`, `Partition` |
| Conditional expectation | `FiniteProbabilitySpace`, `RandomVariable` |
| Filtrations and martingales | `Filtration`, `Martingale`, `StoppingTime`, `StoppedProcess` |
| Time-series analysis | AR, MA, ARMA, ARIMA, SARIMA, correlograms, stationarity tests |
| Box–Jenkins workflow | identification, candidate estimation, validation, deterministic selection, forecasting |
| State-space / Kalman | filtering, smoothing, forecasting, likelihood estimation, innovation diagnostics, adequacy |

The repository preserves explicit numerical contracts and deterministic regression fixtures across the major workflows.

## Installation

```bash
python -m pip install stochx
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
from stochx.stochastic import MarkovChain, empirical_state_frequencies

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

### State-space filtering

```python
from stochx.timeseries import local_level_filter

result = local_level_filter(
    [1.0, 2.0, 3.0],
    process_variance=0.0,
    observation_variance=1.0,
    initial_level=0.0,
    initial_variance=1.0,
)

print(result.states)
print(result.log_likelihood)
```

### Full state-space workflow

```python
import numpy as np
from stochx.timeseries import run_local_level_workflow

result = run_local_level_workflow(
    np.array([1.0, 1.2, np.nan, 1.3, 1.4, 1.25, 1.5, 1.55]),
    diagnostic_lags=2,
    alpha=0.10,
    forecast_steps=3,
)

print(result.smoother.smoothed_state)
print(result.forecast.forecast)
```

## Public API

The public stochastic namespace is available from `stochx.stochastic`, while time-series and state-space functionality is exposed from `stochx.timeseries`.

The detailed reference is maintained in the [API documentation](https://faycal214.github.io/stochx/).

## Examples

Every major mathematical area has runnable examples. The examples directory includes focused stochastic-process examples as well as the time-series and state-space workflows.

```text
examples/
├── 01_discrete_markov_chain.py
├── 02_poisson_process.py
├── 03_continuous_markov_chain.py
├── 04_birth_death_process.py
├── 05_conditional_expectation.py
├── 06_martingale.py
├── 07_api_operations.py
├── 08_eviews_time_series_workflow.py
├── 09_state_space_kalman.py
├── 10_state_space_workflow.py
└── api_quickstart.py
```

CI executes every `examples/*.py` script and builds the documentation strictly.

## Documentation

The documentation site separates:

- **Course material** for mathematical development.
- **Package / API** for public Python objects and validation rules.
- **Time Series** for the USTHB-style analysis workflow and state-space extensions.
- **Worked Examples** for executable end-to-end usage.
- **Release readiness** for distribution, migration, and release-surface requirements.

Start at the [documentation site](https://faycal214.github.io/stochx/).

For users upgrading from the `0.2.x` line, see the [0.2.x migration guide](https://faycal214.github.io/stochx/stage12/migration_0_2_x/).

## Development and quality gates

The repository uses GitHub Actions to run the full test suite on Python 3.10, 3.11, and 3.12. The CI pipeline also checks:

- public API docstring coverage;
- API-reference page coverage;
- documentation structure;
- runnable example coverage;
- package identity;
- strict MkDocs builds;
- source and wheel distribution builds;
- Twine metadata validation;
- clean-wheel installation/import verification.

Run the main suite locally with:

```bash
pytest -q tests --disable-warnings
```

Run all examples locally with:

```bash
for example in examples/*.py; do
    echo "=== $example ==="
    python "$example" >/dev/null
done
```

Build and validate distributions locally before a release:

```bash
python -m build
python -m twine check dist/*
```

## Versioning

StochX follows semantic versioning for public API changes:

- `MAJOR` for incompatible public API changes;
- `MINOR` for backwards-compatible features;
- `PATCH` for backwards-compatible fixes.

The package version is defined once in `stochx/__init__.py` and is used by the build configuration, avoiding separate version values that can drift.

## Release status

StochX is currently in the release-hardening phase following the frozen Stage 8–11 numerical contracts. The current package version remains `0.2.0` until a release-candidate decision explicitly selects the next published version.

PyPI publishing is prepared through a tag-based release workflow; publication requires the repository's PyPI trusted publisher configuration.

## License

MIT
