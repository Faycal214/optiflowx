# Course → API Map

Use this page to move from a time-series course concept to the concrete StochX object that implements it.

| Course concept | StochX | Guide |
|---|---|---|
| workfile / sample | `Workfile` | [EViews workflow](time-series/eviews-workflow.md) |
| time-series object | `TimeSeries` | [Data and series](time-series/data-series.md) |
| lag / lead | `Y(-1)` / `Y(1)` | [Transformations](time-series/transforms.md) |
| difference / log difference | `D(Y)` / `DLOG(Y)` | [Transformations](time-series/transforms.md) |
| smoothing / decomposition | `moving_average`, `decompose`, `holt`, `holt_winters` | [Transformations](time-series/transforms.md) |
| DF / ADF | `dickey_fuller`, `adf`, `dickey_fuller_sequential` | [Stationarity](time-series/stationarity.md) |
| KPSS / Phillips–Perron | `kpss_test`, `phillips_perron` | [Stationarity](time-series/stationarity.md) |
| ACF / PACF | `acf`, `pacf` | [Correlation](time-series/correlation.md) |
| correlogram | `correlogram` | [Correlation](time-series/correlation.md) |
| OLS / equation | `Workfile.ls`, `ols` | [EViews workflow](time-series/eviews-workflow.md) |
| AR / MA / ARMA | `fit_ar`, `fit_ma`, `fit_arma` | [Models](time-series/models.md) |
| ARIMA / SARIMA | `fit_arima`, `fit_sarima` | [Models](time-series/models.md) |
| ARMA errors | `AR(...)`, `MA(...)`, `parse_error_terms` | [EViews workflow](time-series/eviews-workflow.md) |
| residual diagnostics | `ljung_box`, `breusch_godfrey`, `jarque_bera`, `arch_test` | [Diagnostics](time-series/diagnostics.md) |
| Box–Jenkins | identification, candidate estimation, validation, selection and forecast helpers | [Box–Jenkins](time-series/box-jenkins.md) |
| forecast | `forecast_box_jenkins`, `naive_forecast`, `drift_forecast` | [Forecasting](time-series/forecasting.md) |
| prediction interval | `prediction_interval` | [Forecasting](time-series/forecasting.md) |
| state-space | `LinearStateSpace` | [State-space](time-series/state-space.md) |
| Kalman filtering / smoothing | `kalman_filter`, `kalman_smoother` | [State-space](time-series/state-space.md) |
