# Time Series — USTHB and EViews-style methodology

This page is the course-oriented reference for the applied time-series layer. It describes the **methodology and reasoning** behind the Python workflow; the [Time Series User Guide](time-series/index.md) provides the practical module-by-module instructions.

## 1. From raw data to a model

The workflow is deliberately sequential:

```text
raw series
→ workfile / sample
→ descriptive analysis
→ transformations
→ stationarity
→ ACF/PACF
→ model specification
→ estimation
→ residual diagnostics
→ selection
→ forecast
```

This is close to the workflow EViews users already know, but the full analysis is now a Python script that can be versioned and rerun.

## 2. Workfile thinking

```python
from stochx.timeseries import Workfile

wf = Workfile.from_csv(
    "macro.csv",
    date_column="DATE",
    frequency="M",
)
wf.set_sample("2010-01-01 2024-12-01")
```

A workfile contains named series and a current estimation sample. This is the central object for multi-series empirical analysis.

## 3. Expressions and transformations

The expression layer keeps common EViews conventions familiar:

```python
wf.eval("GDP(-1)")
wf.eval("D(GDP)")
wf.eval("DLOG(CPI)")
wf.eval("LOG(IP)")
wf.generate("GDP_GROWTH", "DLOG(GDP)")
```

Use transformations to express a modeling assumption explicitly: remove trend/level behavior, stabilize variance, construct growth rates or create lagged regressors.

## 4. Stationarity

A unit-root decision must identify the deterministic specification first. StochX exposes DF/ADF, KPSS and Phillips–Perron procedures.

```python
from stochx.timeseries import adf

result = adf(wf["GDP"], regression="c", lags=1, autolag=None)
print(result.summary())
print(result.interpret())
```

The test output should be read together with the series plot and economic context.

## 5. Identification with ACF/PACF

Once a working stationary series has been selected, inspect the autocorrelation and partial autocorrelation functions.

```python
from stochx.timeseries import correlogram

corr = correlogram(wf["GDP_GROWTH"].values, nlags=24)
print(corr.table())
```

Use ACF/PACF as model-identification evidence, not as a mechanical order generator.

## 6. Model families

StochX covers:

- AR(p);
- MA(q);
- ARMA(p,q);
- ARIMA(p,d,q);
- SARIMA(p,d,q) × (P,D,Q,s);
- regression equations with ARMA error terms.

For example:

```python
from stochx.timeseries import fit_arima

result = fit_arima(wf["GDP"], order=(1, 1, 1))
print(result.summary())
```

## 7. Box–Jenkins

For systematic candidate comparison, use the explicit workflow:

```python
from stochx.timeseries import (
    identify_box_jenkins,
    estimate_box_jenkins_candidates,
    validate_box_jenkins_candidates,
    select_box_jenkins_model,
    forecast_box_jenkins,
)

ident = identify_box_jenkins(wf["GDP"], d=1, nlags=24, max_p=3, max_q=3)
estimation = estimate_box_jenkins_candidates(wf["GDP"], ident.candidate_orders)
validation = validate_box_jenkins_candidates(estimation, lags=12, alpha=0.05)
selection = select_box_jenkins_model(validation, criterion="aic")
forecast = forecast_box_jenkins(selection, steps=12, alpha=0.05)
```

The important distinction is that **estimation success is not model adequacy**. Residual validation is a separate decision.

## 8. Regression equations and ARMA errors

EViews-style equation specifications remain textual:

```python
eq = wf.ls(
    "TBILL C LOG(M1) LOG(CPI) LOG(IP) @TREND AR(1) MA(1)",
    name="EQ20",
)
```

`C` is the intercept, `@TREND` is the deterministic trend term where supported, and `AR(n)` / `MA(n)` refer to the disturbance process.

## 9. Diagnostics

Residual validation includes serial-correlation, normality and variance diagnostics where the selected model supports them. Typical tools include Ljung–Box, Breusch–Godfrey, Jarque–Bera, Kolmogorov–Smirnov, Breusch–Pagan and ARCH tests.

A report should state the null hypothesis, statistic, p-value and modeling consequence.

## 10. Forecasting

Forecast reports should include point forecasts and uncertainty. StochX keeps standard errors, lower/upper bounds, horizon and future index together when supported.

## 11. State-space extension

The linear-Gaussian state-space layer provides an alternative representation when latent states, missing observations or recursive dynamics matter.

The workflow is:

```text
model → Kalman filter → RTS smoother → innovation diagnostics → adequacy → forecast
```

See the [state-space guide](time-series/state-space.md).

## 12. EViews migration principle

The migration target is not a GUI imitation. The target is **methodological familiarity**:

- the same sequence of statistical decisions;
- recognizable names and expressions;
- table-oriented results;
- explicit estimation samples;
- reproducible scripts;
- documented numerical conventions.

Where exact EViews benchmark outputs are available in the repository, they are treated as frozen numerical fixtures rather than informal claims of universal parity.
