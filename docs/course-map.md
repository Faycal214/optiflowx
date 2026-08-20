# Course → API Map

Use this page when moving from a mathematical course concept to the concrete StochX object that implements it.

## Time Series and Econometrics

| Concept | StochX | Guide |
|---|---|---|
| workfile / sample | `Workfile` | [EViews workflow](time-series/eviews-workflow.md) |
| named indexed series | `TimeSeries` | [Data and series](time-series/data-series.md) |
| lag / lead | `Y(-1)`, `Y(1)`, `Workfile.lag` | [Transformations](time-series/transforms.md) |
| difference / log difference | `D(Y)`, `DLOG(Y)` | [Transformations](time-series/transforms.md) |
| decomposition / smoothing | `decompose`, `moving_average`, `holt`, `holt_winters` | [Decomposition](time-series/transforms.md) |
| ADF / DF | `adf`, `dickey_fuller`, `dickey_fuller_sequential` | [Stationarity](time-series/stationarity.md) |
| KPSS / Phillips–Perron | `kpss_test`, `phillips_perron` | [Stationarity](time-series/stationarity.md) |
| ACF / PACF | `acf`, `pacf` | [Correlation](time-series/correlation.md) |
| correlogram | `correlogram` | [Correlation](time-series/correlation.md) |
| OLS equation | `Workfile.ls`, `ols` | [EViews workflow](time-series/eviews-workflow.md) |
| AR / MA / ARMA | `fit_ar`, `fit_ma`, `fit_arma` | [Models](time-series/models.md) |
| ARIMA / SARIMA | `fit_arima`, `fit_sarima` | [Models](time-series/models.md) |
| ARMA-error equation | `AR(...)`, `MA(...)`, `parse_error_terms` | [EViews workflow](time-series/eviews-workflow.md) |
| residual whiteness | `ljung_box`, `box_pierce`, `residual_correlogram` | [Diagnostics](time-series/diagnostics.md) |
| serial correlation | `breusch_godfrey`, `durbin_watson_test` | [Diagnostics](time-series/diagnostics.md) |
| normality | `jarque_bera`, `normality_ks` | [Diagnostics](time-series/diagnostics.md) |
| heteroskedasticity / ARCH | `breusch_pagan`, `arch_test` | [Diagnostics](time-series/diagnostics.md) |
| Box–Jenkins identification | `identify_box_jenkins` | [Box–Jenkins](time-series/box-jenkins.md) |
| candidate estimation | `estimate_box_jenkins_candidates` | [Box–Jenkins](time-series/box-jenkins.md) |
| residual validation | `validate_box_jenkins_candidates` | [Box–Jenkins](time-series/box-jenkins.md) |
| deterministic selection | `select_box_jenkins_model` | [Box–Jenkins](time-series/box-jenkins.md) |
| forecast | `forecast_box_jenkins`, `naive_forecast`, `drift_forecast` | [Forecasting](time-series/forecasting.md) |
| prediction interval | `prediction_interval` | [Forecasting](time-series/forecasting.md) |
| linear-Gaussian state-space | `LinearStateSpace` | [State-space](time-series/state-space.md) |
| Kalman filtering | `kalman_filter`, `local_level_filter` | [State-space](time-series/state-space.md) |
| RTS smoothing | `kalman_smoother` | [State-space](time-series/state-space.md) |
| state-space forecasting | `kalman_forecast` | [State-space](time-series/state-space.md) |
| local-level likelihood | `estimate_local_level` | [State-space](time-series/state-space.md) |
| innovation diagnostics | `kalman_innovation_diagnostics` | [State-space](time-series/state-space.md) |
| state-space adequacy | `state_space_adequacy` | [State-space](time-series/state-space.md) |

## Stochastic Processes

### Markov chains

| Concept | StochX | Guide |
|---|---|---|
| transition matrix `P` | `MarkovChain` | [Markov chains](stochastic/markov-chains.md) |
| `P^n` | `transition_matrix_at` | [Markov chains](stochastic/markov-chains.md) |
| distribution `mu_0 P^n` | `state_distribution` | [Markov chains](stochastic/markov-chains.md) |
| accessibility | `accessible` | [Markov chains](stochastic/markov-chains.md) |
| communication | `communicate` | [Markov chains](stochastic/markov-chains.md) |
| recurrence / transience | `classify_states` | [Markov chains](stochastic/markov-chains.md) |
| stationarity | `stationary_distribution` | [Markov chains](stochastic/markov-chains.md) |
| hitting / return | corresponding first-passage methods | [Markov chains](stochastic/markov-chains.md) |

### Poisson processes

| Concept | StochX | Guide |
|---|---|---|
| homogeneous process | `PoissonProcess` | [Poisson](stochastic/poisson-processes.md) |
| non-homogeneous intensity | `NonHomogeneousPoissonProcess` | [Poisson](stochastic/poisson-processes.md) |
| counting probabilities | process probability methods | [Poisson](stochastic/poisson-processes.md) |
| arrival simulation | process simulation methods | [Poisson](stochastic/poisson-processes.md) |

### CTMC and birth-death

| Concept | StochX | Guide |
|---|---|---|
| generator `Q` | `ContinuousTimeMarkovChain` | [CTMC](stochastic/ctmc-birth-death.md) |
| transition semigroup | `transition_matrix_at` | [CTMC](stochastic/ctmc-birth-death.md) |
| holding time | CTMC holding-time methods | [CTMC](stochastic/ctmc-birth-death.md) |
| trajectory | `CTMCPath` | [CTMC](stochastic/ctmc-birth-death.md) |
| birth/death rates | `BirthDeathProcess` | [CTMC](stochastic/ctmc-birth-death.md) |

### Probability and conditional expectation

| Concept | StochX | Guide |
|---|---|---|
| finite probability space | `FiniteProbabilitySpace` | [Probability objects](stochastic/probability-objects.md) |
| random variable | `RandomVariable` | [Probability objects](stochastic/probability-objects.md) |
| partition | `Partition` | [Probability objects](stochastic/probability-objects.md) |
| conditional expectation | finite-space conditional operations | [Probability objects](stochastic/probability-objects.md) |

### Martingales

| Concept | StochX | Guide |
|---|---|---|
| filtration | `Filtration` | [Martingales](stochastic/martingales.md) |
| martingale | `Martingale` | [Martingales](stochastic/martingales.md) |
| stopping time | `StoppingTime` | [Martingales](stochastic/martingales.md) |
| stopped process | `StoppedProcess` | [Martingales](stochastic/martingales.md) |
