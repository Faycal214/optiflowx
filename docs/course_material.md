# Course Material

This section is the mathematical reference behind StochX. It is deliberately separate from the Python API: equations, definitions, hypotheses and decision rules are explained here first; implementation details live in the API and User Guide sections.

## Time Series and Econometrics

The project follows the supplied USTHB course material as its primary methodological reference.

### Workflow

```text
observed series
→ descriptive analysis
→ transformations / stationarization
→ deterministic specification
→ ACF / PACF
→ identification
→ estimation
→ validation / testing
→ selection
→ forecasting
```

### Topics

- Time-series data and the distinction between a realization and its generating process.
- Weak/second-order stationarity.
- Deterministic versus stochastic non-stationarity.
- TS and DS processes.
- Random walks and unit roots.
- Differencing and other stationarization methods.
- DF/ADF and related unit-root procedures.
- ACF and PACF estimation and interpretation.
- AR, MA and ARMA processes.
- Stationarity and invertibility conditions.
- Seasonal ARMA / SARMA and ARIMA-family models.
- Box–Jenkins identification.
- Parameter estimation, including likelihood-based estimation.
- Coefficient significance and model validation.
- Residual diagnostics and model adequacy.
- Information criteria such as AIC, BIC/SC and HQC.
- Forecasting and prediction intervals.
- Smoothing/filtering and decomposition.
- Regression models with time-series disturbances.
- Linear-Gaussian state-space representations and Kalman methods.

The [Time Series User Guide](time-series/index.md) translates these topics into the StochX API. The [EViews numerical parity](eviews_parity.md) page records the software-specific conventions being benchmarked.
