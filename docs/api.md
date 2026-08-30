# API Quick Map

StochX exposes one public domain: **time-series analysis and econometrics**.

| Family | Public entry points |
|---|---|
| Data / workfiles | `TimeSeries`, `Workfile` |
| Expressions | `Expression`, `evaluate` |
| Equations | `Equation`, `EquationResult`, `ols` |
| Correlation | `acf`, `pacf`, `correlogram` |
| Stationarity | `adf`, `dickey_fuller`, `kpss_test`, `phillips_perron` |
| Decomposition | `decompose`, `moving_average`, `exponential_smoothing`, `holt`, `holt_winters` |
| AR / MA / ARMA | `fit_ar`, `fit_ma`, `fit_arma` |
| ARIMA / SARIMA | `fit_arima`, `fit_sarima`, `estimate` |
| Box–Jenkins | identification, candidate estimation, validation, selection and forecast helpers |
| Diagnostics | Ljung–Box, Breusch–Godfrey, Jarque–Bera, ARCH, Breusch–Pagan and residual diagnostics |
| Forecasting | `naive_forecast`, `drift_forecast`, `prediction_interval`, `metrics` |
| State-space | `LinearStateSpace` and Kalman filtering/smoothing/forecasting workflows |
| Reports / plotting | result objects, tables, interpretation and `plot_*` helpers |

See the full [Time-series API map](api/time-series.md).
