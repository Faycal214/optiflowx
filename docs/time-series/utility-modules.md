# Supporting time-series modules

The main workflow pages cover the modeling path. This page documents the supporting modules that users often need once the basic workflow is understood.

## Regression

`ols` and `RegressionResult` provide regression calculations used in equation workflows.

```python
from stochx.timeseries import ols

result = ols(y, X, add_intercept=True)
print(result.summary())
```

Use the regression layer when the question is primarily about deterministic regressors; use Box–Jenkins or state-space workflows when serial dynamics are central.

## Forecast metrics

`metrics` summarizes forecast errors and lets you compare models on the same hold-out sample and horizon.

```python
from stochx.timeseries import metrics

report = metrics(actual, predicted)
```

## Theoretical AR/MA utilities

The theory module exposes model-side calculations such as polynomial roots, stationarity/invertibility checks, process means, impulse responses and theoretical autocorrelation.

```python
from stochx.timeseries import is_stationary_ar, impulse_response

print(is_stationary_ar([0.5]))
print(impulse_response([0.5], 10))
```

These functions are especially useful for teaching because they connect the fitted parameters back to the model's theoretical behavior.

## Simulation

`white_noise`, `ar`, `ma`, `arma`, `random_walk` and `sarma` generate synthetic series for exercises and regression testing.

The stochastic namespace also provides simulation for process objects. See [Stochastic simulation](../stochastic/simulation.md) for the reproducibility rules shared by both layers.

## Plotting

The plotting helpers cover the standard workflow views:

- `plot_series` for the raw/indexed series;
- `plot_correlogram` for ACF/PACF diagnostics;
- `plot_eviews_correlogram` for the EViews-style report view;
- `plot_decomposition` for level/trend/seasonal components;
- `plot_forecast` for point forecasts and intervals.

A plotting function should be treated as a view of a numerical result, not as the numerical result itself. The result object remains the source of truth for tests and downstream computation.

## Result and table formatting

`UnifiedResult`, `ResultTable`, `format_correlogram_table` and the summary/table helpers provide consistent report-oriented output.

This design supports both notebooks and automated report generation: the same calculation can be inspected interactively or rendered into a stable table for a script.

## Interpretation

`interpret_correlogram` and the result-level `interpret()` methods turn raw statistics into plain-language decision guidance. Use them as a communication layer; keep the raw statistic and metadata in the analysis record.
