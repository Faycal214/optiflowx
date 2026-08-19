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

## Equations

```python
eq = wf.ls("GDP C CONS CONS(-1)", name="EQ01")
print(eq.summary())
print(eq.table())
```

## Phase B — serial correlation and ARMA error correction

`AR(n)` and `MA(n)` terms describe the **regression disturbance**, not observed regressors. Ranges such as `AR(1 to 2)` and `MA(1 to 3)` expand into the corresponding contiguous error orders.

```python
eq = wf.ls(
    "TBILL C LOG(M1) LOG(CPI) LOG(IP) @TREND AR(1) MA(1)",
    name="EQ20",
)
```

### EViews-compatible ML/BFGS mode

For ARMA-error equations StochX uses Gaussian maximum likelihood with BFGS, stationary/invertible ARMA constraints, the EViews coefficient ordering, and the EViews inverse-root convention. `@TREND` is zero-based within the estimation sample, matching the official tutorial benchmark.

A benchmark may supply the published EViews estimates as `start_params` to isolate the likelihood/parameterization parity from the separate automatic-start-value problem:

```python
result = wf.ls(specification, name="EQ20", start_params=eviews_reference_vector)
```

The result exposes:

```python
result.params       # C, structural coefficients, AR(...), MA(...), SIGMASQ
result.statistics() # AIC, Schwarz, Hannan-Quinn, etc.
result.roots_report()
```

`roots_report()` follows EViews and returns **inverse characteristic roots** rather than the raw polynomial roots.

The official benchmark covers:

```text
EQ18: MA(1)
EQ19: MA(1 to 3)
EQ20: AR(1) + MA(1)
EQ21: AR(1 to 2) + MA(1)
```

The Phase B numerical regression tests assert coefficient-by-coefficient parity, `SIGMASQ`, inverse AR/MA roots, AIC, Schwarz and Hannan-Quinn at the precision displayed by EViews. The EViews OPG covariance convention for standard errors is intentionally tracked as a separate parity target.

## Serial-correlation diagnostics

The equation result exposes a Breusch-Godfrey test:

```python
bg = eq.serial_correlation(lags=1)
```

and the raw LM/F forms are available through `breusch_godfrey_raw`.

## Model results

AR, MA, ARMA, ARIMA and SARIMA results expose a common interface for fitted parameters, residuals, forecasts, information criteria, and diagnostics where supported.

## Stationarity and Stage 7 — DF / ADF

The Dickey-Fuller/ADF implementation follows the course convention of treating deterministic specifications separately.

## Course critical values

Stage 7 carries explicit Dickey-Fuller tables for Models 1, 2 and 3 and does not use ordinary `statsmodels` p-value/critical-value decisions.

## Common ADF lag order and residual whitening

Stage 7 chooses the smallest common `p` whose Model 3 residuals pass the requested Ljung-Box whiteness check, then holds `p` fixed across Models 3, 2 and 1.

## Course-faithful sequential decision tree

The workflow follows the conditional Model 3 → Model 2 → Model 1 strategy, including β/α significance checks and non-standard F3/F2 tests.
