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

## Phase B — serial correlation and ARMA error correction

Phase B adds the EViews tutorial's error-process syntax to the equation layer. `AR(n)` and `MA(n)` terms describe the **regression disturbance**, not observed regressors. Ranges such as `AR(1 to 2)` and `MA(1 to 3)` expand into the corresponding contiguous error orders.

```python
eq = wf.ls(
    "TBILL C LOG(M1) LOG(CPI) LOG(IP) @TREND AR(1) MA(1)",
    name="EQ20",
)
```

For the official EViews Time Series tutorial benchmark, the current Phase B fixture covers:

```text
EQ18: MA(1)
EQ19: MA(1 to 3)
EQ20: AR(1) + MA(1)
EQ21: AR(1 to 2) + MA(1)
```

These use maximum-likelihood estimation with BFGS, matching the estimation method displayed by EViews in the tutorial. The current implementation deliberately treats the numerical optimizer/likelihood convention as a separate parity target; Phase B tests first lock the exact specifications, sample size, parameter names, and error orders.

`@TREND` is supported as a deterministic trend regressor in these equations.

## Serial-correlation diagnostics

The equation result exposes a Breusch-Godfrey test:

```python
bg = eq.serial_correlation(lags=1)
print(bg)
```

The LM and F versions are also available:

```python
from stochx.timeseries import breusch_godfrey_raw

print(breusch_godfrey_raw(eq.result, lags=1))
```

The null hypothesis is no residual serial correlation through the requested order. Durbin-Watson, Box-Pierce and Ljung-Box remain available in the general diagnostic layer.

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

## Course critical values

Stage 7 does not use `statsmodels`' critical-value dictionary to make the DF/ADF decision. StochX carries the course's Dickey-Fuller table values explicitly for Models 1, 2 and 3 at n=50, 100, 250 and asymptotic infinity.

The table-row convention is explicit:

```text
n <= 50      -> n=50 row
50 < n <=100 -> n=100 row
100 < n<=250 -> n=250 row
n > 250      -> asymptotic (n=∞) row
```

This reproduces the course's worked ADF example where 91 effective observations are compared with the 100-observation row. The reported p-value remains informational; the course critical-value inequality is the decision rule.

The public constants are:

```python
from stochx.timeseries import DF_CRITICAL_VALUES, DF_F_CRITICAL_VALUES
```

## Common ADF lag order and residual whitening

The course introduces lagged differences to whiten the innovations and recommends the minimal specification that removes residual autocorrelation. Stage 7 therefore chooses the smallest common `p` whose Model 3 residuals pass a Ljung-Box whiteness check through the requested diagnostic lag horizon.

```python
report = dickey_fuller_sequential(
    wf["GDP"],
    max_lags=8,
    autolag="AIC",      # retained for API compatibility; the sequential workflow uses whitening/parsimony
    whitening_lags=12,
    alpha=0.05,
)
```

The selected `p` is then held fixed for Models 3, 2 and 1. The result reports whether whitening was achieved or whether the maximum permitted lag was used as a fallback. Supplying `autolag=None` keeps an explicitly fixed common lag order for TP work.

## Course-faithful sequential decision tree

The workflow is not simply “stop at the first unit-root rejection”. It follows the conditional specification logic taught in the course:

```text
MODEL 3: constant + trend
       |
       +-- test γ = 0 with Model 3 DF critical values
       |
       +-- reject H0
       |     |
       |     +-- test β = 0 with a standard Student-t critical value
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
       +-- test γ = 0 with Model 2 DF critical values
       |
       +-- reject H0
       |     |
       |     +-- test α = 0 with a standard Student-t critical value
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

The F2/F3 decisions use the USTHB non-standard critical values and deliberately do not use ordinary Fisher p-values.

## Unified Stage 7 results

```python
print(report.table())
print(report.specification_table())
print(report.summary())
print(report.interpret())
```

The main table reports each Model 3/2/1 DF/ADF statistic, p-value, 1%, 5%, and 10% course critical values, lag order, and decision. The specification table reports the conditional trend/constant tests and the non-standard F3/F2 joint tests used by the decision tree.
