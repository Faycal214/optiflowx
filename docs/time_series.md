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

## Box–Jenkins workflow

The public Box–Jenkins surface follows a deterministic identification → estimation → residual validation → model selection → forecasting workflow. The Stage 9 contracts freeze candidate-order generation, auditable estimation metadata, adequacy decisions, deterministic tie-breaking, transformation restoration, and forecast metadata.

```python
from stochx.timeseries import (
    identify_box_jenkins,
    estimate_box_jenkins_candidates,
    validate_box_jenkins_candidates,
    select_box_jenkins_model,
    forecast_box_jenkins,
)

ident = identify_box_jenkins(y, d=1, nlags=12, max_p=2, max_q=2)
estimation = estimate_box_jenkins_candidates(y, ident.candidate_orders)
validation = validate_box_jenkins_candidates(estimation, lags=12, alpha=0.05)
selection = select_box_jenkins_model(validation)
forecast = forecast_box_jenkins(selection, steps=12, alpha=0.05)
```

## State-space and Kalman filtering

Stage 10 adds a public linear-Gaussian state-space core without changing the Stage 8 correlogram or Stage 9 Box-Jenkins contracts.

For the scalar local-level model:

```python
from stochx.timeseries import local_level_filter

result = local_level_filter(
    [1.0, 2.0, 3.0],
    process_variance=0.0,
    observation_variance=1.0,
    initial_level=0.0,
    initial_variance=1.0,
)

print(result.states)
print(result.log_likelihood)
```

For a general linear-Gaussian model:

```python
import numpy as np
from stochx.timeseries import LinearStateSpace, kalman_filter

model = LinearStateSpace(
    transition=np.eye(2),
    design=np.eye(2),
    state_cov=np.eye(2) * 0.1,
    observation_cov=np.eye(2),
    initial_state=np.zeros(2),
    initial_cov=np.eye(2),
)

result = kalman_filter(
    np.array([
        [1.0, 2.0],
        [np.nan, 3.0],
        [np.nan, np.nan],
        [4.0, 5.0],
    ]),
    model,
)
```

Missing values are handled per observed scalar dimension. A partially missing row updates only with finite dimensions; an all-missing row performs prediction only and contributes no likelihood increment. `nobs` counts time rows, while `effective_nobs` counts observed scalar measurements and `missing_observations` counts missing scalar measurements.

`KalmanFilterResult` exposes filtered/predicted states and covariances, innovations, innovation covariances, Gaussian log likelihood, observation accounting, and the observed-dimension mask. Its numerical arrays are immutable.

## Stage 11 state-space extension

Stage 11 extends the filtering core into a complete auditable workflow without changing the Stage 10 filtering contract.

### Smoothing

```python
from stochx.timeseries import kalman_smoother

smoothed = kalman_smoother(y, model, filter_result=result)
print(smoothed.smoothed_state)
```

`kalman_smoother` applies a deterministic Rauch–Tung–Striebel backward recursion and retains all time rows, including fully missing observations.

### Forecasting

```python
from stochx.timeseries import kalman_forecast

future = kalman_forecast(y, model, steps=5, alpha=0.10, filter_result=result)
print(future.forecast)
print(future.lower)
print(future.upper)
```

Forecast intervals are returned with immutable forecast, standard-error, lower-bound, and upper-bound arrays.

### Likelihood estimation

```python
from stochx.timeseries import estimate_local_level

estimate = estimate_local_level(y, initial_level=0.0, initial_variance=1.0)
print(estimate.process_variance)
print(estimate.observation_variance)
print(estimate.aic, estimate.bic)
```

### Innovation diagnostics and adequacy

```python
from stochx.timeseries import kalman_innovation_diagnostics, state_space_adequacy

diag = kalman_innovation_diagnostics(result)
adequacy = state_space_adequacy(diag, lags=4, alpha=0.05)
print(diag.standardized_innovations)
print(adequacy.adequate)
```

### End-to-end workflow

```python
from stochx.timeseries import run_local_level_workflow

workflow = run_local_level_workflow(
    y,
    diagnostic_lags=4,
    alpha=0.05,
    forecast_steps=8,
)
```

The workflow composes estimation, filtering, smoothing, innovation diagnostics, adequacy checks, and forecasting around one canonical filtering result. The runnable reference is `examples/10_state_space_workflow.py`.

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
result.params
result.statistics()
result.roots_report()
result.covariance_method
result.covariance_matrix()
result.bse
result.tvalues
result.pvalues
```

For ARMA maximum-likelihood equations, the covariance matrix is formed from the inverse of the sum of per-observation score outer products. The Phase B benchmark uses this OPG information matrix directly, matching the published EViews equation output.

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
