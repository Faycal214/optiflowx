# State-space and Kalman filtering

State-space models represent an observed time series as a noisy measurement of an underlying latent state.

A linear-Gaussian model can be written as

$$x_t = F x_{t-1} + \eta_t, \qquad y_t = Hx_t + \varepsilon_t,$$

with Gaussian state noise $\eta_t$ and observation noise $\varepsilon_t$.

## When to use a state-space model

Use the state-space interface when:

- the quantity of interest is latent;
- the dynamics are naturally recursive;
- observations can be partially missing;
- you need filtered and smoothed states;
- forecast uncertainty should propagate through the state covariance.

## General model

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

result = kalman_filter(observations, model)
```

The filter returns predicted and filtered states/covariances, innovations, innovation covariances, Gaussian log likelihood and observation-accounting metadata.

## Missing observations

Missingness is handled per scalar observation dimension. A partial row updates only with observed components; an all-missing row performs prediction without an observation update.

This preserves the time dimension instead of compressing the series by dropping missing rows.

## Local-level shortcut

For a scalar level model, the convenience interface is simpler:

```python
from stochx.timeseries import local_level_filter

result = local_level_filter(
    y,
    process_variance=0.2,
    observation_variance=1.0,
    initial_level=0.0,
    initial_variance=1.0,
)
```

## Smoothing

Filtering uses information available up to time $t$. Smoothing revisits past states after observing the full sample.

```python
from stochx.timeseries import kalman_smoother

smoothed = kalman_smoother(y, model, filter_result=result)
```

StochX uses a deterministic Rauch–Tung–Striebel backward recursion.

## Forecasting

```python
from stochx.timeseries import kalman_forecast

future = kalman_forecast(
    y,
    model,
    steps=5,
    alpha=0.10,
    filter_result=result,
)
```

## Full workflow

The integrated local-level workflow combines estimation, filtering, smoothing, innovation diagnostics, adequacy testing and forecasting:

```python
from stochx.timeseries import run_local_level_workflow

workflow = run_local_level_workflow(
    y,
    diagnostic_lags=4,
    alpha=0.05,
    forecast_steps=8,
)
```

For the complete result layout, see `StateSpaceWorkflowResult` in the API reference and `examples/10_state_space_workflow.py`.

## Numerical contract

State-space results are protected by deterministic regression fixtures. Output arrays are copied/immutable at the public boundary so later user mutation cannot alter a previously computed result.
