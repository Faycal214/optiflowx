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

## Stationarity

```python
from stochx.timeseries import adf

result = adf(wf["GDP"], regression="ct", lags=2, autolag=None)
print(result.summary())
print(result.interpret())
```

The deterministic specifications follow the course convention: no constant, constant, and constant plus trend. The sequential workflow is available through `dickey_fuller_sequential`.
