# Forecasting

Forecasting should be the end of a validated modeling workflow, not the first operation.

## Simple baselines

StochX exposes baseline forecasts that are useful for sanity checks:

```python
from stochx.timeseries import naive_forecast, drift_forecast, metrics

naive = naive_forecast(y, steps=12)
drift = drift_forecast(y, steps=12)
```

Use these before celebrating an elaborate model. A complex model should beat a defensible baseline for the metric that matters.

## Prediction intervals

For stochastic forecasts, a point prediction is incomplete without uncertainty information.

```python
from stochx.timeseries import prediction_interval

lower, upper = prediction_interval(
    forecast,
    standard_error,
    alpha=0.05,
)
```

The `alpha` argument controls the significance level, so a 95% interval uses `alpha=0.05`.

## Box–Jenkins forecasts

```python
from stochx.timeseries import forecast_box_jenkins

result = forecast_box_jenkins(selection, steps=12, alpha=0.05)
print(result.forecast)
print(result.standard_error)
print(result.lower)
print(result.upper)
```

The forecast result keeps the model order, criterion, horizon, alpha level and output index together with the numerical arrays.

## State-space forecasts

```python
from stochx.timeseries import kalman_forecast

result = kalman_forecast(
    observations,
    model,
    steps=8,
    alpha=0.05,
    filter_result=filtered,
)
```

State-space forecasting propagates the filtered state and covariance through future transition steps. This is especially useful when the observation process and latent state are not identical.

## Evaluation

Use held-out observations whenever possible:

```python
from stochx.timeseries import metrics

report = metrics(actual, predicted)
print(report)
```

Evaluation should be tied to the forecast horizon and the decision problem. Never compare models on a metric that ignores the scale or cost of forecast errors in the application.

## Index preservation

When the original series carries a datetime index with a valid frequency, StochX preserves that temporal structure for future labels. This is essential for report-ready forecasts and downstream joins.

## Interpretation

A useful forecast report communicates:

- the model and estimation sample;
- the horizon;
- point forecasts;
- standard errors or intervals;
- the future index;
- any transformations that were restored before reporting.
