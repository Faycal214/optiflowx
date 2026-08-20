# StochX

[![PyPI](https://img.shields.io/pypi/v/stochx)](https://pypi.org/project/stochx/)
[![Python](https://img.shields.io/pypi/pyversions/stochx)](https://pypi.org/project/stochx/)
[![CI](https://github.com/Faycal214/stochx/actions/workflows/test.yml/badge.svg)](https://github.com/Faycal214/stochx/actions/workflows/test.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-1081C2?style=flat)](https://faycal214.github.io/stochx/)
[![License](https://img.shields.io/github/license/stochx/stochx)](LICENSE)

**StochX is a lightweight Python time-series and stochastic-process library built for analysts who want an EViews-like workflow in Python.**

The main applied focus is **time-series analysis, econometrics and forecasting**. The package keeps familiar EViews ideas—workfiles, samples, series expressions, lags, differences, `C`, `@TREND`, equations, ADF, ARMA errors, correlograms, model reports and forecasts—while making every step executable, inspectable and reproducible in Python.

A separate stochastic-process layer covers the mathematical foundations of Markov chains, Poisson processes, CTMCs, birth-death processes, probability spaces and martingales.

## Why StochX?

| Need | StochX approach |
|---|---|
| Move an EViews workflow into Python | EViews-inspired workfiles, expressions and equation syntax |
| Follow a standard time-series methodology | data → transformation → stationarity → ACF/PACF → estimation → diagnostics → selection → forecasting |
| Keep analysis reproducible | scripts instead of GUI state, explicit parameters and deterministic tests |
| Get report-ready results | `summary()`, `table()`, `interpret()`, roots, diagnostics and forecast objects |
| Work from course material | mathematical explanations remain separate from the software/API reference |
| Go beyond ARIMA | linear-Gaussian state-space models, Kalman filtering/smoothing and innovation diagnostics |
| Learn stochastic processes | explicit mathematical objects with validated state spaces and trajectories |

StochX is **EViews-inspired rather than a binary-compatible EViews clone**. Numerical parity is treated as an explicit benchmark wherever a reference fixture exists, so users can see which conventions are verified.

## The time-series workflow

```text
Workfile / data
      ↓
Inspect and describe
      ↓
Generate lags, differences, logs, trend terms
      ↓
Decompose / smooth when useful
      ↓
ADF / KPSS / PP stationarity decisions
      ↓
ACF / PACF / correlogram
      ↓
AR / MA / ARMA / ARIMA / SARIMA
      ↓
Residual diagnostics
      ↓
Box–Jenkins validation + deterministic selection
      ↓
Forecast + prediction intervals
```

The state-space workflow extends the same idea:

```text
Linear state-space model
      ↓
Kalman filter
      ↓
RTS smoother
      ↓
Innovation diagnostics
      ↓
Adequacy tests
      ↓
Forecasting
```

## Quick EViews-style example

```python
from stochx.timeseries import Workfile, adf, estimate

wf = Workfile.from_csv("macro.csv", date_column="DATE", frequency="M")
wf.set_sample("2010-01-01 2024-12-01")

print(wf.info())
print(wf.eval("GDP(-1)"))
print(wf.generate("DGDP", "D(GDP)").summary())

eq = wf.ls("GDP C CONS CONS(-1)", name="EQ01")
print(eq.summary())
print(eq.interpret())

unit_root = adf(wf["GDP"], regression="c", lags=1, autolag=None)
print(unit_root.summary())
```

## Box–Jenkins in Python

```python
from stochx.timeseries import (
    identify_box_jenkins,
    estimate_box_jenkins_candidates,
    validate_box_jenkins_candidates,
    select_box_jenkins_model,
    forecast_box_jenkins,
)

ident = identify_box_jenkins(y, d=1, nlags=24, max_p=3, max_q=3)
estimation = estimate_box_jenkins_candidates(y, ident.candidate_orders)
validation = validate_box_jenkins_candidates(estimation, lags=12, alpha=0.05)
selection = select_box_jenkins_model(validation, criterion="aic")
forecast = forecast_box_jenkins(selection, steps=12, alpha=0.05)
```

## State-space example

```python
import numpy as np
from stochx.timeseries import run_local_level_workflow

workflow = run_local_level_workflow(
    np.array([1.0, 1.2, np.nan, 1.3, 1.4, 1.25, 1.5]),
    diagnostic_lags=4,
    alpha=0.05,
    forecast_steps=3,
)

print(workflow.smoother.smoothed_state)
print(workflow.forecast.forecast)
```

## Installation

```bash
python -m pip install stochx
```

Development and documentation environments:

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[docs]"
```

## Documentation

The documentation is organized by **what you are trying to do** rather than by release stage:

- **Time Series** — the main applied guide, written as a workflow and optimized for EViews users moving to Python.
- **Stochastic Processes** — a separate mathematical guide for Markov chains, Poisson processes, CTMCs, probability objects and martingales.
- **Course Material** — mathematical foundations, notation, hypotheses and worked derivations.
- **Package / API** — exact public objects, parameters, properties, methods and numerical conventions.
- **Examples** — runnable scripts corresponding to the guide pages.

Start with the [Time Series User Guide](https://faycal214.github.io/stochx/time-series/).

## Supported time-series areas

```text
Data / Workfile
Series expressions and transformations
Descriptive statistics
Smoothing and decomposition
ADF / DF / KPSS / Phillips–Perron
ACF / PACF / correlograms
OLS and EViews-style equations
AR / MA / ARMA / ARIMA / SARIMA
ARMA-error regression
Breusch–Godfrey / Ljung–Box / Jarque–Bera / ARCH / variance tests
Box–Jenkins identification → estimation → validation → selection → forecast
Prediction intervals and forecast metrics
Linear-Gaussian state-space models
Kalman filtering / smoothing / forecasting
Local-level likelihood estimation
Innovation diagnostics and state-space adequacy
```

## Supported stochastic-process areas

```text
Discrete-time Markov chains
Poisson and non-homogeneous Poisson processes
Continuous-time Markov chains
Birth-death processes
Finite probability spaces
Random variables and partitions
Conditional expectation
Filtrations and martingales
Stopping times and stopped processes
Simulation and trajectory analysis
```

## Examples

Runnable examples live in `examples/` and are executed in CI. The most relevant applied examples are:

```text
08_eviews_time_series_workflow.py
09_state_space_kalman.py
10_state_space_workflow.py
```

## Quality and reproducibility

StochX treats numerical conventions as part of the public contract. The repository includes deterministic regression fixtures, missing-observation tests, model-selection contracts and CI checks across Python 3.10, 3.11 and 3.12.

```bash
pytest -q tests --disable-warnings
```

All example scripts can be executed with:

```bash
for example in examples/*.py; do
    echo "=== $example ==="
    python "$example" >/dev/null
done
```

Build and validate distributions with:

```bash
python -m build
python -m twine check dist/*
```

## Version

StochX follows semantic versioning. The current release is **0.3.0**.

The numerical and public API contracts introduced during Stages 8–11 are frozen. Future breaking changes should follow a deliberate major-version review.

## License

MIT
