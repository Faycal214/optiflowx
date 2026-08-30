# API Reference

StochX exposes one public domain: `stochx.timeseries`.

The API Reference documents the exact Python objects used for time-series analysis, econometrics, Box–Jenkins workflows, forecasting, diagnostics and state-space models.

## Main API families

- `TimeSeries`, `Workfile`, `Expression`, `Equation`
- `acf`, `pacf`, `correlogram`
- `adf`, `dickey_fuller`, `kpss_test`, `phillips_perron`
- smoothing, decomposition and seasonal transformations
- `fit_ar`, `fit_ma`, `fit_arma`, `fit_arima`, `fit_sarima`
- Box–Jenkins identification, estimation, validation, selection and forecasting
- residual and specification diagnostics
- forecast utilities and prediction intervals
- `LinearStateSpace` and the Kalman filter/smoother/forecasting workflow
- result tables, interpretation and EViews-oriented reports

The [Time-series API map](time-series.md) provides the course-to-code mapping. For methodology and mathematical foundations, use [Course Material](../course_material.md) and the [Time Series User Guide](../time-series/index.md).
