# API Quick Map

StochX exposes two public namespaces. The **time-series namespace is the primary applied API**; the stochastic namespace contains the mathematical process layer.

## Time Series

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
| Box–Jenkins | `identify_box_jenkins`, `estimate_box_jenkins_candidates`, `validate_box_jenkins_candidates`, `select_box_jenkins_model`, `forecast_box_jenkins` |
| Diagnostics | `ljung_box`, `breusch_godfrey`, `jarque_bera`, `arch_test`, `breusch_pagan`, residual diagnostics |
| Forecasting | `naive_forecast`, `drift_forecast`, `prediction_interval`, `metrics`, `restore_differences` |
| State-space | `LinearStateSpace`, `kalman_filter`, `kalman_smoother`, `kalman_forecast`, `estimate_local_level`, innovation diagnostics, adequacy, workflow |
| Plotting / reports | `plot_*`, `summary()`, `table()`, `interpret()` |

See the full [Time-series API map](api/time-series.md).

## Stochastic Processes

| Domain | Reference |
|---|---|
| DTMC | [`MarkovChain`](api/markov_chain.md) |
| Poisson / NHPP | [`PoissonProcess`](api/poisson_process.md) |
| CTMC | [`ContinuousTimeMarkovChain`](api/continuous_time_markov_chain.md) |
| CTMC trajectories | [`CTMCPath`](api/ctmc_path.md) |
| Birth-death | [`BirthDeathProcess`](api/birth_death_process.md) |
| Probability space | [`FiniteProbabilitySpace`](api/finite_probability_space.md) |
| Random variable | [`RandomVariable`](api/random_variable.md) |
| Partition | [`Partition`](api/partition.md) |
| Filtration | [`Filtration`](api/filtration.md) |
| Martingale | [`Martingale`](api/martingale.md) |
| Stopping time | [`StoppingTime`](api/stopping_time.md) |
| Stopped process | [`StoppedProcess`](api/stopped_process.md) |

See the full [Stochastic-process API map](api/stochastic-processes.md).

## Choosing where to start

- Need to analyze economic, business or scientific measurements over time? Start with [Time Series](time-series/index.md).
- Need a Markov chain, counting process or martingale for probability modeling? Start with [Stochastic Processes](stochastic/index.md).
- Need mathematical derivations first? Start with [Course Material](course_material.md).

The API documents the Python contract. The User Guide explains the workflow. The Course Material explains the mathematics.
