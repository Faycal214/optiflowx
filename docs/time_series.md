# Time Series

StochX Time Series is organized around the USTHB workflow used for time-series coursework: data preparation, stationarisation, identification, estimation, validation, and forecasting.

## Workfile

```python
from stochx.timeseries import Workfile

wf = Workfile()
wf.add("GDP", gdp_values)
wf.add("CONS", consumption_values)
wf.set_sample(10, 100)
```

A workfile keeps named series and a current estimation sample.

## EViews-style expressions

```python
wf.eval("GDP")
wf.eval("GDP(-1)")
wf.eval("GDP(1)")
wf.eval("D(GDP)")
wf.eval("DLOG(GDP)")
wf.eval("LOG(GDP)")
wf.eval("GDP(-1) + 0.5*CONS")
```

Generated expressions can be stored back into the workfile:

```python
wf.generate("D_GDP", "D(GDP)")
wf.generate("LGDP", "LOG(GDP)")
```

Differenced expressions retain the workfile length with leading missing observations, matching the workfile-oriented time-series convention.

## Equations

An OLS equation can be written in the familiar EViews-style form:

```python
eq = wf.ls("GDP C CONS CONS(-1)", name="EQ01")

print(eq.summary())
print(eq.table())
print(eq.interpret())
```

`C` denotes the constant. Lags and other StochX expressions can be used as regressors.

## Model results

AR, MA, ARMA, ARIMA and SARIMA results expose a common interface:

```python
result = estimate(wf["GDP"], p=1, d=1, q=1)

print(result.summary())
print(result.table())
print(result.interpret())
```

The result layer exposes coefficients, standard errors, t-statistics, p-values, information criteria, fitted values, residuals, forecasts, stability checks, and residual diagnostics where supported by the model.

## Stationarity and Stage 7 — DF / ADF

The Dickey-Fuller/ADF implementation follows the course convention of treating deterministic specifications separately:

```text
Model 3: constant + deterministic trend
Model 2: constant, no trend
Model 1: no constant, no trend
```

The ADF regression tests the coefficient on the lagged level:

```text
H0: gamma = 0   -> unit root / non-stationarity
H1: gamma < 0   -> stationarity under the selected deterministic specification
```

Example:

```python
from stochx.timeseries import adf, dickey_fuller_sequential

model3 = adf(wf["GDP"], regression="ct", lags=2, autolag=None)
print(model3.summary())
print(model3.table())
print(model3.interpret())
```

### Non-standard Dickey-Fuller critical values

The decision is **not** made by comparing the ADF statistic with an ordinary Student-t or normal critical value. StochX uses the critical values associated with the selected deterministic specification and compares:

```text
Reject H0 when ADF statistic < the corresponding DF critical value.
```

The reported p-value is retained for reference, but it is explicitly labelled informational and is not used to replace the course's critical-value decision rule.

### Sequential workflow

```python
report = dickey_fuller_sequential(
    wf["GDP"],
    max_lags=2,
    autolag=None,
    alpha=0.05,
)

print(report.summary())
print(report.table())
print(report.interpret())
```

The workflow evaluates Model 3 first. If the unit-root null is not rejected, it proceeds to Model 2; if it is still not rejected, it proceeds to Model 1. Each model keeps its own regression-specific critical values and decision rule.

The unified sequential table contains the test statistic, information p-value, 1%, 5%, and 10% critical values, and the decision at each specification.
