# StochX

## A practical time-series library for Python

StochX is a lightweight scientific Python library with two clear layers:

- **Time Series** — the main applied layer for econometrics, forecasting and EViews-style analysis.
- **Stochastic Processes** — a separate mathematical layer for probability and stochastic-process coursework.

The time-series documentation is the best place to start for analysts who are moving an existing EViews workflow into Python.

## What makes the time-series layer different?

StochX keeps the workflow familiar:

```text
workfile → sample → series expressions → stationarity
        → ACF/PACF → model estimation → diagnostics
        → model selection → forecast → report
```

The implementation uses Python-native objects, but the vocabulary is intentionally recognizable: `Workfile`, `C`, `@TREND`, `D(Y)`, `DLOG(Y)`, `Y(-1)`, `wf.ls(...)`, ADF, AR/MA terms, correlograms, result tables and interpretation helpers.

The project does not claim to be a universal EViews clone. Instead, it makes the **same statistical workflow easy to reproduce in Python**, and it records exact numerical conventions in regression tests where parity has been benchmarked.

## Start with the guides

### Time Series

Read the [Time Series User Guide](time-series/index.md) first. It covers the complete applied workflow from data preparation to forecasting, plus the state-space/Kalman extension.

### Stochastic Processes

Use the [Stochastic Processes User Guide](stochastic/index.md) for discrete-time Markov chains, Poisson processes, CTMCs, birth-death processes, probability objects, martingales and simulation.

### Course Material

The [Course Material](course_material.md) section separates mathematical explanations, notation and assumptions from the software API.

### Package / API

The [Time-series API map](api/time-series.md) and [Stochastic-process API map](api/stochastic-processes.md) explain the public Python surface. The individual API pages contain the exact object-level reference.

## Installation

```bash
python -m pip install stochx
```

## Minimal time-series example

```python
from stochx.timeseries import Workfile, adf

wf = Workfile.from_csv("macro.csv", date_column="DATE", frequency="M")
wf.set_sample("2010-01-01 2024-12-01")

print(wf.info())
print(wf.eval("GDP(-1)"))
print(wf.generate("DGDP", "D(GDP)").summary())

eq = wf.ls("GDP C CONS CONS(-1)", name="EQ01")
print(eq.summary())

print(adf(wf["GDP"], regression="c", lags=1, autolag=None).summary())
```

## Quality

The package is tested across Python 3.10, 3.11 and 3.12. The repository also runs all examples and validates documentation builds and distribution artifacts.

The current stable package version is **0.3.0**.
