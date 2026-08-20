# Time Series User Guide

StochX is primarily designed to make **practical time-series analysis feel familiar to EViews users while remaining a native Python library**.

The time-series guide follows the workflow most users already know:

1. load or create a workfile;
2. inspect and describe the data;
3. generate transformations and lags;
4. decompose or smooth the series when appropriate;
5. test stationarity and identify deterministic terms;
6. inspect ACF/PACF and the correlogram;
7. estimate AR, MA, ARMA, ARIMA and SARIMA models;
8. validate residuals and model assumptions;
9. compare candidate models with transparent selection rules;
10. forecast and report prediction intervals.

The same sequence also leads naturally to the linear state-space and Kalman workflow.

## Why StochX for an EViews user?

StochX deliberately uses familiar EViews concepts and names where the underlying operation is implemented: **workfiles, samples, series expressions, lags, differences, `C`, `@TREND`, equations, LS estimation, ADF, AR/MA error terms, summary tables, interpretation helpers, roots reports, and forecast reports**.

This means that an EViews workflow can be translated into Python without replacing the user's statistical methodology with a completely different framework.

The design goal is **EViews-inspired workflow compatibility**, not a hidden black-box wrapper. The implementation is Python-native, testable, scriptable, and reproducible. Where StochX includes published or frozen numerical benchmarks, those are treated as explicit regression contracts.

## A complete workflow in one page

```python
import numpy as np
from stochx.timeseries import (
    Workfile,
    adf,
    correlogram,
    identify_box_jenkins,
    estimate_box_jenkins_candidates,
    validate_box_jenkins_candidates,
    select_box_jenkins_model,
    forecast_box_jenkins,
)

wf = Workfile.from_dataframe(dataframe, frequency="M")
wf.set_sample(0, wf.nobs - 1)

series = wf["Y"]
print(series.describe())

unit_root = adf(series, regression="c", lags=1, autolag=None)
print(unit_root.summary())

corr = correlogram(series.values, nlags=24)
print(corr.table())

ident = identify_box_jenkins(series, d=1, nlags=24, max_p=3, max_q=3)
estimation = estimate_box_jenkins_candidates(series, ident.candidate_orders)
validation = validate_box_jenkins_candidates(estimation, lags=12, alpha=0.05)
selection = select_box_jenkins_model(validation)
forecast = forecast_box_jenkins(selection, steps=12, alpha=0.05)
```

The purpose of each step is explained in the pages below rather than hidden inside one convenience function.

## Reading the guide

Each method page follows the same structure:

- **What it does** — practical goal and when to use it.
- **Mathematical idea** — model, statistic, or hypothesis being computed.
- **Syntax** — the StochX call and its important parameters.
- **Worked example** — a small runnable example.
- **Reading the output** — how to interpret tables and diagnostics.
- **Failure modes** — common mistakes and what StochX reports.
- **API reference** — exact public objects and related functions.

This is intentionally close to the style of the scikit-learn User Guide: concept first, then implementation details, examples, and references rather than a list of signatures without context. citeturn866057view0turn866057search0

## Time-series map

| Topic | Start here |
|---|---|
| Workfiles and EViews syntax | [EViews-style workflow](eviews-workflow.md) |
| Data and `TimeSeries` objects | [Data and series](data-series.md) |
| Lags, differences, logs and deterministic terms | [Transformations](transforms.md) |
| Smoothing and seasonal structure | [Decomposition](transforms.md) |
| DF / ADF, KPSS and PP | [Stationarity](stationarity.md) |
| ACF, PACF and correlograms | [Correlation](correlation.md) |
| AR, MA, ARMA, ARIMA, SARIMA | [Models](models.md) |
| Box–Jenkins methodology | [Box–Jenkins](box-jenkins.md) |
| Residual tests and adequacy | [Diagnostics](diagnostics.md) |
| Forecasts and prediction intervals | [Forecasting](forecasting.md) |
| Linear-Gaussian state-space and Kalman | [State-space](state-space.md) |
| EViews-style result tables and interpretation | [Reports and interpretation](reports.md) |
| Supporting modules: regression, theory, simulation, plotting, result tables | [Supporting modules](utility-modules.md) |

## Design principles

StochX time-series functionality is built around five principles:

1. **Course-faithful terminology.** Definitions, notation, hypotheses and decision rules remain explicit.
2. **EViews familiarity.** Common commands and expressions are intentionally recognizable.
3. **Report-oriented output.** Results expose statistics, tables, diagnostics, coefficients and interpretations rather than returning an opaque model object alone.
4. **Deterministic behavior.** Frozen fixtures protect numerical conventions and regression stability.
5. **Composable Python.** Every step can be called independently, inspected, tested and embedded in a larger workflow.
