# Box–Jenkins and EViews automatic ARIMA

## Two workflows

StochX exposes two distinct workflows because EViews does.

### Manual Box–Jenkins identification

```python
from stochx.timeseries import identify_box_jenkins

id_result = identify_box_jenkins(
    y,
    d=1,
    nlags=24,
    max_p=2,
    max_q=2,
)
```

This workflow uses the stationary series, ACF/PACF and significant-lag patterns to produce a small candidate set for human inspection. It is the identification workflow for the traditional Box–Jenkins procedure.

### EViews automatic ARIMA

EViews' `autoarma` procedure performs:

1. dependent-variable transformation selection;
2. successive KPSS tests for differencing;
3. exhaustive AR/MA/seasonal AR/seasonal MA order enumeration up to the configured maxima;
4. model selection by AIC, SIC, HQ or MSE;
5. optional inclusion of non-converged models.

EViews documents these steps explicitly. citeturn131363view0turn131363view1

StochX now exposes:

```python
from stochx.timeseries import autoarma

result = autoarma(
    y,
    max_diff=2,
    max_ar=4,
    max_ma=4,
    max_sar=0,
    max_sma=0,
    select="aic",
)
```

The defaults match EViews:

| Option | StochX default | EViews default |
|---|---:|---:|
| max differencing | 2 | 2 |
| max AR | 4 | 4 |
| max MA | 4 | 4 |
| max SAR | 0 | 0 |
| max SMA | 0 | 0 |
| selection | AIC | AIC |
| KPSS significance | 5% | 5% |
| seasonal periodicity | workfile-dependent | observations/year |
| non-converged models | excluded | excluded |

EViews documents these defaults in the `autoarma` series procedure. citeturn131363view0

## Transformation selection

With `tform="auto"`, StochX implements EViews' published None/Log rule:

```text
D(y_t)^2 = alpha_1 + beta_1 y_t
D(log(y_t))^2 = alpha_2 + beta_2 log(y_t)
```

The log specification is preferred when the absolute t-statistic on `beta_2` is smaller than that on `beta_1`. EViews documents this as its automatic None/Log heteroskedasticity rule. citeturn580441search20

```python
autoarma(y, tform="auto")
autoarma(y, tform="none")
autoarma(y, tform="log")
```

Auto and Log require strictly positive observations.

Box-Cox transformation is an EViews option but is not yet included in the certified automatic path.

## Differencing selection

EViews uses successive KPSS tests:

```text
d = 0
  ↓ reject stationarity null
d = 1
  ↓ reject
d = 2
  ↓ stop
```

The procedure stops at the first differencing order for which the KPSS stationarity null is not rejected, or at the configured maximum. citeturn463747search7

The complete KPSS decision history is preserved in:

```python
result.kpss_history
result.differencing_order
```

## ARMA candidate grid

EViews estimates every AR/MA combination up to the maximum order, with optional seasonal AR/SMA dimensions. Information criteria are then used only within the same transformation/differencing scale. citeturn131363view1

The StochX result table contains:

```text
p d q P D Q
LogLik
AIC
SIC
HQ
converged
included
error
```

## Selection

```python
result.selected_order
result.selected
result.table()
```

The default is AIC, matching EViews' `autoarma` default. SIC and HQ are supported explicitly.

MSE-based selection and SAIC/BMA forecast averaging are EViews features not yet included in this implementation.

## Forecasting

The selected fitted object remains accessible:

```python
forecast = result.forecast(12)
```

This keeps the automatic selection result connected to the same EViews-style equation/model result object used elsewhere in the package.

## Relationship to the rest of StochX

```text
Workfile
  ↓
Transformation
  ↓
KPSS differencing
  ↓
ARMA candidate grid
  ↓
EViews ML estimation
  ↓
AIC/SIC/HQ
  ↓
Selected equation
  ↓
Diagnostics
  ↓
Forecast
```

This is intentionally different from the manual Box–Jenkins path:

```text
Workfile
  ↓
Stationary series
  ↓
ACF / PACF
  ↓
Human identification
  ↓
Candidate equations
  ↓
Diagnostics
  ↓
Final selection
```

## Exact-parity boundary

The EViews automatic procedure, defaults, transformation rule, KPSS differencing rule, candidate-grid structure and selection criterion are now represented. Exact numerical equality for every candidate model still requires captured EViews `autoarma` output fixtures, especially for:

- automatic transformation decisions;
- non-convergence handling;
- exact EViews starting values;
- forecast intervals;
- MSE selection;
- SAIC/BMA averaging;
- Box-Cox transformation.
