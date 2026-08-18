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
Model 3: ΔYt = α + βt + γYt-1 + ΣφiΔYt-i + εt
Model 2: ΔYt = α + γYt-1 + ΣφiΔYt-i + εt
Model 1: ΔYt = γYt-1 + ΣφiΔYt-i + εt
```

The ADF root test is:

```text
H0: γ = 0   -> unit root / non-stationarity
H1: γ < 0   -> stationarity under the selected deterministic specification
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

The decision is **not** made by comparing the DF/ADF statistic with an ordinary Student-t or normal critical value. StochX uses the critical values associated with the selected deterministic specification and compares:

```text
Reject H0 when the DF/ADF statistic < the corresponding non-standard DF critical value.
```

The reported p-value is retained for reference, but is explicitly labelled informational and is not used to replace the course's critical-value decision rule.

### Common ADF lag order

The course first chooses the number of lagged differences required to control autocorrelation in the innovations. StochX selects that lag order once and then applies the same `p` to Models 3, 2, and 1.

```python
report = dickey_fuller_sequential(
    wf["GDP"],
    max_lags=8,
    autolag="AIC",
    alpha=0.05,
)
```

With `autolag=None`, `max_lags` is treated as the fixed common lag order. With an information criterion, the order is selected on Model 3 and then held fixed for the sequential specification tests.

### Course-faithful sequential decision tree

The workflow is not simply “stop at the first unit-root rejection”. It follows the conditional specification logic taught in the course:

```text
MODEL 3: constant + trend
       |
       +-- test γ = 0 with DF critical values
       |
       +-- reject H0
       |     |
       |     +-- test β = 0 with a standard two-sided critical value
       |           |
       |           +-- β significant -> retain Model 3 / TS
       |           +-- β not significant -> continue to Model 2
       |
       +-- do not reject H0
             |
             +-- test H3,0: γ = 0 and β = 0 with non-standard F3 critical values
                   |
                   +-- reject H3,0 -> integrated Model 3 case
                   +-- do not reject -> continue to Model 2

MODEL 2: constant
       |
       +-- test γ = 0 with DF critical values
       |
       +-- reject H0
       |     |
       |     +-- test α = 0 with a standard two-sided critical value
       |           |
       |           +-- α significant -> retain Model 2 / TS
       |           +-- α not significant -> continue to Model 1
       |
       +-- do not reject H0
             |
             +-- test H2,0: γ = 0 and α = 0 with non-standard F2 critical values
                   |
                   +-- reject H2,0 -> integrated Model 2 case
                   +-- do not reject -> continue to Model 1

MODEL 1: no constant, no trend
       |
       +-- test γ = 0 with Model 1 DF critical values
       +-- reject -> stationary around zero
       +-- do not reject -> difference-stationary / integrated candidate
```

The joint F decisions deliberately do **not** use ordinary Fisher p-values. StochX uses the non-standard F2/F3 critical values from the USTHB course tables and reports their source in the result object.

### Unified Stage 7 results

```python
print(report.table())
print(report.specification_table())
print(report.summary())
print(report.interpret())
```

The main table reports each Model 3/2/1 DF/ADF statistic, p-value, 1%, 5%, and 10% critical values, lag order, and decision. The specification table reports the conditional trend/constant tests and the non-standard F3/F2 joint tests used by the decision tree.
