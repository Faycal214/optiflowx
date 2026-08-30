# StochX

## EViews-inspired time-series analysis in Python

StochX is a focused Python library for time-series analysis, econometrics and forecasting, designed around the workflow taught in the USTHB time-series courses and the conventions familiar to EViews users.

The project is deliberately narrow: time series only. It does not provide a general stochastic-process, probability, or simulation framework.

## Workflow

    workfile / data
    -> sample and series expressions
    -> transformations / smoothing / decomposition
    -> stationarity and unit-root analysis
    -> ACF / PACF / correlogram
    -> model identification
    -> estimation
    -> residual validation
    -> model selection
    -> forecasting
    -> reports

## Main interfaces

The public API is centered on stochx.timeseries:

- Workfile and TimeSeries for data/workfile management
- EViews-style expressions such as Y(-1), D(Y), DLOG(Y), LOG(Y), C and @TREND
- ACF, PACF and correlograms
- DF/ADF, KPSS and Phillips-Perron stationarity procedures
- OLS and EViews-style equation specifications
- AR, MA, ARMA, ARIMA and SARIMA models
- residual and specification diagnostics
- Box-Jenkins identification, estimation, validation, selection and forecasting
- smoothing/decomposition and forecast utilities
- linear-Gaussian state-space and Kalman workflows
- report-oriented result objects and tables

## EViews compatibility

StochX is implemented natively in Python. The compatibility target is the analysis workflow and numerical conventions, not a graphical clone. Where an EViews result is available as a benchmark, the repository records it as an explicit regression fixture.

See EViews numerical parity and the Time Series User Guide.

## Course material

The course material documents the mathematical and methodological foundations separately from the API. The supplied USTHB material covers stationary and non-stationary processes, ARMA identification, estimation, validation, forecasting and the associated regression/time-series methodology.

## Installation

    python -m pip install stochx

## Minimal example

    from stochx.timeseries import Workfile, adf

    wf = Workfile.from_csv("macro.csv", date_column="DATE", frequency="M")
    wf.set_sample("2010-01-01 2024-12-01")

    print(wf.info())
    print(wf.eval("GDP(-1)"))
    print(wf.generate("DGDP", "D(GDP)").summary())

    model = wf.ls("GDP C CONS CONS(-1)", name="EQ01")
    print(model.summary())
    print(adf(wf["GDP"], regression="c", lags=1, autolag=None).summary())

## Quality

Numerical behavior is tested through ordinary unit tests and explicit EViews-parity fixtures. The release workflow also validates distributions, documentation and executable examples.
