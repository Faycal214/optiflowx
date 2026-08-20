# API Reference

StochX has two public namespaces with different purposes:

- `stochx.timeseries` — the main applied layer for time-series analysis, econometrics, Box–Jenkins workflows, forecasting and state-space models.
- `stochx.stochastic` — mathematical stochastic-process objects for probability and stochastic-process coursework.

The API pages use the same pattern as the User Guide: what the object represents, constructor parameters, important properties, public methods, a small example and numerical/validation conventions.

## Time-series API

Start with the [Time-series API map](time-series.md). The most important public families are:

- `TimeSeries`, `Workfile`, `Expression`, `Equation`;
- `acf`, `pacf`, `correlogram`;
- `adf`, `dickey_fuller`, `kpss_test`, `phillips_perron`;
- `moving_average`, `exponential_smoothing`, `holt`, `holt_winters`, `decompose`;
- `fit_ar`, `fit_ma`, `fit_arma`, `fit_arima`, `fit_sarima`, `estimate`;
- Box–Jenkins identification, estimation, validation, selection and forecasting;
- residual diagnostics and statistical tests;
- naive/drift forecasting and forecast metrics;
- `LinearStateSpace` and the Kalman filter/smoother/forecasting/diagnostics workflow.

## Stochastic-process API

Start with the [Stochastic-process API map](stochastic-processes.md). The object families are:

- `MarkovChain`;
- `PoissonProcess` and `NonHomogeneousPoissonProcess`;
- `ContinuousTimeMarkovChain`, `CTMCPath`, `BirthDeathProcess`;
- `FiniteProbabilitySpace`, `RandomVariable`, `Partition`;
- `Filtration`, `Martingale`, `StoppingTime`, `StoppedProcess`;
- trajectory-frequency analysis and exceptions.

## API conventions

Public constructors validate mathematical invariants early. Result objects preserve the metadata required to interpret a calculation, and numerical arrays are protected from accidental mutation where the corresponding contract requires immutability.

The package does not hide failures behind automatic fallbacks. A rejected model specification, invalid matrix, impossible probability vector or unavailable adequate model is surfaced explicitly.

For mathematical definitions and proofs, use [Course Material](../course_material.md). For applied workflows, start with [Time Series](../time-series/index.md) or [Stochastic Processes](../stochastic/index.md).
