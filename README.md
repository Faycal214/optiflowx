# StochX

[![PyPI](https://img.shields.io/pypi/v/stochx)](https://pypi.org/project/stochx/)
[![Python](https://img.shields.io/pypi/pyversions/stochx)](https://pypi.org/project/stochx/)
[![CI](https://github.com/Faycal214/stochx/actions/workflows/test.yml/badge.svg)](https://github.com/Faycal214/stochx/actions/workflows/test.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-1081C2?style=flat)](https://faycal214.github.io/stochx/)
[![License](https://img.shields.io/github/license/stochx/stochx)](LICENSE)

**StochX is an EViews-inspired Python library for time-series analysis, econometrics, and forecasting.**

The project is deliberately focused on **time series only**. It is designed to reproduce the workflow, logic, syntax, calculations, conventions, and report-oriented outputs familiar to EViews users while remaining native, scriptable, inspectable, and reproducible in Python.

## Why StochX?

| Need | StochX approach |
|---|---|
| Move an EViews workflow into Python | EViews-inspired workfiles, samples, expressions and equation syntax |
| Follow the university methodology | course-faithful stationarity, identification, estimation, validation and forecasting |
| Reproduce calculations | explicit numerical conventions and regression fixtures |
| Get report-ready results | summaries, tables, diagnostics, roots and forecast objects |
| Keep analysis reproducible | scripts instead of GUI state |
| Work beyond basic ARIMA | seasonal models and linear-Gaussian state-space/Kalman workflows |

## Time-series workflow

```text
Workfile / data
      ↓
Inspect and describe
      ↓
Generate lags, differences, logs, trend terms
      ↓
Smooth / decompose when appropriate
      ↓
Stationarity and deterministic specification
      ↓
ACF / PACF / correlogram
      ↓
AR / MA / ARMA / ARIMA / SARIMA
      ↓
Residual diagnostics and validation
      ↓
Model selection
      ↓
Forecast + prediction intervals
      ↓
EViews-style report
```

## Quick EViews-style example

```python
from stochx.timeseries import Workfile, adf

wf = Workfile.from_csv("macro.csv", date_column="DATE", frequency="M")
wf.set_sample("2010-01-01 2024-12-01")

print(wf.info())
print(wf.eval("GDP(-1)"))
print(wf.generate("DGDP", "D(GDP)").summary())

eq = wf.ls("GDP C CONS CONS(-1)", name="EQ01")
print(eq.summary())

unit_root = adf(wf["GDP"], regression="c", lags=1, autolag=None)
print(unit_root.summary())
```

## Box–Jenkins

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

## Documentation

The documentation is organized around the time-series workflow:

- **Time Series** — practical user guide.
- **Course Material** — mathematical definitions, assumptions, and methodology.
- **EViews numerical parity** — explicit software-comparison benchmarks and conventions.
- **Package / API** — exact public Python interface.
- **Examples** — runnable workflows.

Start with the [Time Series User Guide](https://faycal214.github.io/stochx/time-series/).

## Supported areas

```text
Workfiles / samples / indexed series
EViews-style expressions and transformations
Descriptive statistics
Smoothing and decomposition
DF / ADF / KPSS / Phillips–Perron
ACF / PACF / correlograms
OLS and EViews-style equations
AR / MA / ARMA / ARIMA / SARIMA
ARMA-error regression
Serial-correlation / normality / heteroskedasticity diagnostics
Box–Jenkins identification → estimation → validation → selection → forecast
Prediction intervals and forecast metrics
Linear-Gaussian state-space models
Kalman filtering / smoothing / forecasting
EViews-oriented reports and result tables
```

## Installation

```bash
python -m pip install stochx
```

For development:

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[docs]"
```

## Quality

StochX treats numerical conventions as part of its public contract. The repository contains deterministic regression fixtures, EViews-parity tests, missing-observation tests, model-selection contracts, and release validation across Python 3.10–3.12.

```bash
pytest -q tests --disable-warnings
```

Build distributions with:

```bash
python -m build
python -m twine check dist/*
```

## Version

Current release: **0.3.0**

## License

MIT
