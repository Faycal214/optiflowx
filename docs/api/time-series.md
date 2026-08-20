# Time-series API

The public `stochx.timeseries` namespace is the main applied-analysis surface of StochX.

## Data and workfiles

- `TimeSeries` — named/indexed univariate series.
- `Workfile` — named series, estimation sample and EViews-inspired expression/equation interface.
- `Expression`, `evaluate` — lag/lead/transformation expressions.

## Correlation and correlograms

- `acf`, `pacf`
- `correlogram`
- `format_correlogram_table`, `format_correlogram`
- `interpret_correlogram`

## Stationarity

- `adf`, `dickey_fuller`, `dickey_fuller_sequential`
- `kpss_test`, `phillips_perron`
- `difference`, `trend_regression`, `classify_ts_ds`

## Decomposition and smoothing

- `moving_average`, `weighted_moving_average`
- `exponential_smoothing`, `holt`, `holt_winters`
- `decompose`, `seasonal_difference`, `seasonal_indices`
- `fisher_seasonality_test`

## Model estimation

- `fit_ar`, `fit_ma`, `fit_arma`, `fit_arima`, `fit_sarima`
- `estimate`
- `ols`, `RegressionResult`
- `parse_error_terms`, `ErrorProcess`

## Box–Jenkins

- `identify_box_jenkins`
- `estimate_box_jenkins_candidates`
- `validate_box_jenkins_candidates`
- `select_box_jenkins_model`
- `forecast_box_jenkins`

Each result retains the metadata needed to audit the step rather than exposing only the final coefficients.

## Diagnostics

- `durbin_watson_test`
- `breusch_godfrey`, `breusch_godfrey_raw`
- `ljung_box`, `box_pierce`
- `jarque_bera`, `normality_ks`
- `breusch_pagan`, `arch_test`
- `mean_zero_test`, `variance_ratio_test`
- `residual_correlogram`, `residual_diagnostics`
- `roots_report`, `redundancy_check`

## Forecasting

- `naive_forecast`, `drift_forecast`
- `prediction_interval`
- `metrics`
- `restore_differences`

## State-space

- `LinearStateSpace`
- `kalman_filter`, `local_level_filter`
- `kalman_smoother`
- `kalman_forecast`
- `estimate_local_level`
- `kalman_innovation_diagnostics`
- `state_space_adequacy`
- `run_local_level_workflow`

## Plots and reports

- `plot_series`
- `plot_correlogram`, `plot_eviews_correlogram`
- `plot_decomposition`
- `plot_forecast`
- result objects with `summary()`, `table()` and `interpret()` where provided.

For the conceptual explanation, start with the [Time Series User Guide](../time-series/index.md). For exact stochastic-process object signatures, use the existing class pages in `docs/api/`.
