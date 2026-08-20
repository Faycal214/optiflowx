# Time Series Course Material — USTHB

This page brings the time-series mathematical course material into the StochX documentation so that the theory, the EViews-inspired workflow, and the Python API can be read together.

**Source material:** F. Hamdi, *Analyse de régression et prévision*, USTHB, 11 May 2022, Chapitre 5 — *Modèles de base en séries chronologiques*.

The source develops the notions of temporal data, smoothing and decomposition, forecasting, stationarity, white noise, autocorrelation, the lag operator, MA, AR, ARMA, non-stationary TS/DS processes, ARIMA, ADF, SARIMA and seasonal testing, parameter estimation, and Box–Jenkins validation.

## 1. What is a time series?

A temporal or chronological dataset is a collection of observations indexed by time. The source distinguishes regularly and irregularly spaced observations and focuses on the regularly spaced case. A series may be univariate or multivariate.

A time series is a finite observed trajectory of an underlying stochastic process. The course therefore separates the **process** that generates the observations from the **observed series** that we actually analyse.

Typical examples used in the course include production, sales, exchange rates, stock-market indices, unemployment and industrial production.

## 2. Main objectives of time-series treatment

The course identifies several purposes:

- **Smoothing / filtering:** remove undesirable variations, including irregularity and seasonality.
- **Decomposition:** separate trend-cycle, seasonal and irregular/error components.
- **Modelling:** describe the observed phenomenon with a mathematical time-series model and identify the stochastic process that may have generated the data.
- **Analysis:** interpret the fitted model and its characteristics.
- **Forecasting:** estimate future values and, in some cases, use forecasts to detect errors in a series.
- **Control:** use a variable that can be acted on to influence another variable.

The source also distinguishes deterministic and stochastic modelling and, for stochastic modelling, emphasizes identifying the generating process.

## 3. Forecasting: notation, horizon and information

Let the observed series be

$$y_1,y_2,\ldots,y_n.$$

The forecast origin is `n`, the horizon is `h`, and the future value is denoted by `y_{n+h}`. A forecast is written as `\hat y_{n+h}`, with forecast error

$$e_{n+h}=y_{n+h}-\hat y_{n+h}.$$

A one-step forecast error is the residual associated with the corresponding forecast.

The choice of forecasting method depends on the **information set**: the information currently available and intentionally used for the forecast. The course distinguishes extrapolative methods, explanatory regression methods, and larger systemic/econometric approaches.

Forecasts may be classified by:

- type: event, event date, or quantitative value;
- horizon: short, medium or long term, with the meaning depending on the application;
- nature: point forecast, distributional forecast, or interval forecast.

## 4. Forecast-error criteria

The source introduces several criteria for evaluating forecast errors. For errors `e_1,\ldots,e_n`, the main quantities are:

$$\bar e=\frac1n\sum_{i=1}^n e_i,$$

$$\operatorname{MSE}=\frac1n\sum_{i=1}^n e_i^2,$$

$$\operatorname{RMSE}=\sqrt{\frac1n\sum_{i=1}^n e_i^2},$$

$$\operatorname{MAE}=\frac1n\sum_{i=1}^n|e_i|,$$

and the mean absolute percentage error used in the source for strictly positive variables,

$$\operatorname{MAPE}=\frac1n\sum_{i=1}^n\frac{|e_i|}{|y_i|}.$$

The course notes the familiar trade-offs: squared-error criteria penalize large errors strongly and are less robust to outliers, while absolute-error criteria are more robust.

In StochX, these ideas connect directly to the forecast metrics and prediction-interval objects in the public time-series API.

## 5. Stationarity, white noise and dependence

Classical time-series analysis in the course is organized around three central ideas:

- stationarity;
- autocorrelation;
- white noise.

The source defines a **white-noise process** as a centered sequence of uncorrelated random variables with constant variance. Independence is not required by that definition.

Stationarity is treated at two levels:

### Strict stationarity

The full finite-dimensional distribution is invariant under a common time shift.

### Second-order stationarity

The process has finite second moments, constant mean, and autocovariance depending only on the lag:

$$\operatorname{Cov}(Y_t,Y_{t+h})=\gamma_h.$$

Under second-order stationarity, the variance is constant and equal to `\gamma_0`.

The autocorrelation function is

$$\rho_h=\frac{\gamma_h}{\gamma_0}.$$

The empirical ACF is computed from the observed series and displayed through the empirical correlogram.

The course also introduces the **partial autocorrelation function (PACF)**, including its matrix-based definition and its role in model identification.

## 6. Wold representation

The source presents the Wold theorem as the conceptual foundation for linear stationary modelling. A second-order stationary process can be represented as an infinite moving-average component driven by white noise plus a deterministic component orthogonal to the shocks.

In simplified notation,

$$Y_t=\sum_{j=0}^{\infty}\psi_j\varepsilon_{t-j}+k_t,$$

with `\psi_0=1`, square-summable coefficients, and white-noise innovations.

The practical message for StochX is that finite MA, AR and ARMA models can be read as structured finite representations or approximations of a broader stationary linear process.

## 7. Lag operator and differences

The course uses the lag/backshift operator `L`,

$$LY_t=Y_{t-1},$$

so that

$$L^2Y_t=Y_{t-2},$$

and polynomial expressions in `L` provide a compact notation for time-series recursions.

The first difference is

$$\Delta Y_t=(1-L)Y_t=Y_t-Y_{t-1},$$

while the second difference is

$$\Delta^2Y_t=(1-L)^2Y_t.$$

For a seasonal period `S`, the course introduces

$$\Delta_S Y_t=(1-L^S)Y_t=Y_t-Y_{t-S}.$$

These operators are the theoretical basis for StochX expressions such as `D(X)`, `DLOG(X)`, lag expressions such as `X(-1)`, and seasonal transformations used in ARIMA/SARIMA workflows.

## 8. Moving-average models — MA(q)

The course defines an MA(q) process by

$$Y_t=\mu+\varepsilon_t+\sum_{j=1}^{q}\theta_j\varepsilon_{t-j},$$

or equivalently

$$Y_t=\mu+\Theta(L)\varepsilon_t.$$

Because the innovations are centered, `\mu` is the mean of the process. The source states that an MA(q) is always stationary.

### Invertibility

Invertibility means that the innovation can be expressed as a measurable function of the present and past observations. For the MA(1), the course shows that `|\theta_1|<1` gives the familiar infinite-AR representation.

More generally, an MA(q) is invertible when the roots of its MA polynomial have modulus strictly greater than one.

### ACF of MA(q)

The source derives the MA(q) autocovariance and emphasizes the key identification property:

$$\rho_h=0 \qquad \text{for }h>q.$$

Thus, if an empirical ACF becomes statistically negligible after a finite lag, an MA model of the corresponding order is a natural candidate.

The course also introduces Bartlett's large-sample approximation for the sampling variability of the empirical ACF beyond the MA cutoff.

## 9. Autoregressive models — AR(p)

The source defines an AR(p) process through

$$Y_t-\sum_{j=1}^{p}\phi_jY_{t-j}=c+\varepsilon_t,$$

or

$$\Phi(L)Y_t=c+\varepsilon_t,$$

where the innovations are white noise.

Unlike the MA case, the constant `c` is not directly the process mean. Under stationarity,

$$\mu=\frac{c}{\Phi(1)}.$$

### Causality and stationarity

The course defines causality through a measurable representation of `Y_t` in terms of current and past innovations. The AR(p) process is causal/stationary when the roots of the AR polynomial lie strictly outside the unit circle.

For AR(1), the source obtains

$$\rho_h=\phi_1^h,$$

so the ACF decays exponentially, with possible oscillation when `\phi_1<0`.

### Yule–Walker equations

The source develops the Yule–Walker system for AR(p), connecting the AR coefficients to the autocovariance function. This gives a direct route from theoretical second-order structure to AR parameter estimation.

### Levinson–Durbin

The source presents the Levinson–Durbin recursion for solving the successive AR coefficient systems efficiently from autocorrelation information.

### PACF cutoff

For an AR(p), the PACF is zero from lag `p+1` onward. This is one of the principal theoretical identification rules used with empirical correlograms.

## 10. ARMA(p,q)

The mixed ARMA model combines both types of dynamics:

$$\Phi(L)Y_t=c+\Theta(L)\varepsilon_t.$$

The source highlights a parsimony argument: the limits of purely AR and purely MA representations motivate the mixed class.

Under stationarity,

$$\mu=\frac{c}{1-\phi_1-\cdots-\phi_p}.$$

The roots of the AR polynomial determine stationarity/causality, while the roots of the MA polynomial determine invertibility.

If the two polynomials have no common roots, the course gives the corresponding AR(∞) and MA(∞) representations. It also describes the qualitative ACF/PACF behavior of ARMA processes: depending on the relative orders `p` and `q`, the correlogram may show a finite initial pattern followed by damped exponential/sinusoidal decay rather than an exact finite cutoff.

## 11. Non-stationary processes: TS and DS

When the mean, variance or autocovariances depend on time, the source first asks for a transformation that restores stationarity.

Two important sources are distinguished:

### TS — trend stationary

A TS process has the form

$$Y_t=f(t)+X_t,$$

where `X_t` is stationary and `f(t)` is deterministic.

### DS — difference stationary

A process is integrated of order `d`, `I(d)`, when differencing `d` times produces a stationary process; equivalently, the AR polynomial contains `d` unit roots in the factorization used by the course.

A random walk is the canonical `I(1)` example. The source emphasizes the persistence of shocks for difference-stationary processes.

## 12. ARIMA(p,d,q)

The course defines ARIMA as an ARMA model applied after `d` ordinary differences:

$$\Phi(L)(1-L)^dY_t=c+\Theta(L)\varepsilon_t.$$

The three orders have a direct interpretation:

- `p`: autoregressive order;
- `d`: number of ordinary differences needed for stationarity;
- `q`: moving-average order.

The practical modelling sequence is therefore to identify the required transformation first, then identify the stationary dynamics of the transformed series.

## 13. Dickey–Fuller and ADF

The source presents the Dickey–Fuller family as the principal tool for assessing a unit root.

It uses three deterministic specifications:

### Model 1

$$\Delta Y_t=\pi Y_{t-1}+\sum_{j=1}^{p}\xi_j\Delta Y_{t-j}+\nu_t.$$

### Model 2

$$\Delta Y_t=c+\pi Y_{t-1}+\sum_{j=1}^{p}\xi_j\Delta Y_{t-j}+\nu_t.$$

### Model 3

$$\Delta Y_t=c+\beta t+\pi Y_{t-1}+\sum_{j=1}^{p}\xi_j\Delta Y_{t-j}+\nu_t.$$

The ADF extension adds enough lagged differences to whiten the innovations. The source describes a two-step strategy: choose a lag order able to remove serial correlation, then apply the sequential DF decision strategy to the three deterministic specifications.

StochX implements this course-oriented decision structure in its stationarity API and exposes the resulting specification, statistics and decisions.

## 14. Seasonal models — SARIMA and seasonality testing

For seasonal data, the course introduces the seasonal ARIMA class

$$\Phi_p(L)\Phi_P(L^S)(1-L)^d(1-L^S)^D Y_t
=\Theta_q(L)\Theta_Q(L^S)\varepsilon_t.$$

The seasonal period `S` is explicit in the model.

The source also uses an ANOVA/Fisher approach to test whether the seasonal component contributes significantly. The seasonal-effect hypothesis is tested through a ratio of the period variance to the residual variance, with a rejection rule based on the corresponding F critical value.

This gives a theoretical complement to visual inspection: seasonality should not be inferred from the plot alone when a formal test is appropriate.

## 15. Estimation of ARMA parameters

The course distinguishes estimation based on fixed blocks of observations from online/progressive estimation. For ARMA models with known orders, it presents:

- Yule–Walker (YW);
- ordinary least squares (OLS/MCO) in settings where applicable;
- maximum likelihood (ML/MV).

The source notes that maximum likelihood provides the best precision among these approaches for stationary Gaussian ARMA modelling when its assumptions are appropriate, while Yule–Walker is especially convenient for pure AR models because it reduces the problem to a linear system.

StochX exposes corresponding estimation paths through its AR/MA/ARMA/ARIMA/SARIMA models and the dedicated Box–Jenkins estimation API.

## 16. Box–Jenkins methodology

The source closes with a model-building workflow centred on Box–Jenkins methodology:

1. identify a plausible model class and orders;
2. estimate the model parameters;
3. validate the candidate model;
4. compare/select among adequate candidates;
5. use the final model for forecasting.

Validation includes stability, parameter significance, stationarity and invertibility, and residual adequacy. The residuals should behave like white noise; the source explicitly names the Box–Ljung test, the mean-zero test, Jarque–Bera and other diagnostics as members of the validation battery.

This is the theoretical foundation of the StochX deterministic Box–Jenkins API:

```python
ident = identify_box_jenkins(y, d=1, nlags=24, max_p=3, max_q=3)
estimation = estimate_box_jenkins_candidates(y, ident.candidate_orders)
validation = validate_box_jenkins_candidates(estimation, lags=12, alpha=0.05)
selection = select_box_jenkins_model(validation)
forecast = forecast_box_jenkins(selection, steps=12, alpha=0.05)
```

## 17. How this course maps to StochX

| Course concept | StochX entry point |
|---|---|
| Temporal series / observations | `TimeSeries`, `Workfile` |
| Lag operator | `GDP(-1)` / `Workfile.eval()` |
| Ordinary difference | `D(GDP)` / `Workfile.diff()` |
| Seasonal difference | seasonal transformation utilities |
| ACF / PACF | `acf()`, `pacf()`, `correlogram()` |
| DF / ADF / KPSS / PP | `adf()`, `dickey_fuller()`, `kpss_test()`, `phillips_perron()` |
| AR / MA / ARMA | `fit_ar()`, `fit_ma()`, `fit_arma()`, `estimate()` |
| ARIMA / SARIMA | `fit_arima()`, `fit_sarima()` |
| Box–Jenkins identification | `identify_box_jenkins()` |
| Box–Jenkins estimation | `estimate_box_jenkins_candidates()` |
| Residual validation | `validate_box_jenkins_candidates()` and diagnostic tests |
| Deterministic selection | `select_box_jenkins_model()` |
| Forecasts | `forecast_box_jenkins()`, `prediction_interval()` |
| State-space extension | `LinearStateSpace`, `kalman_filter()`, `kalman_smoother()`, `kalman_forecast()` |

## 18. Learning path

For a reader following the course methodology, the recommended order is:

**Data → transformations → stationarity → ACF/PACF → AR/MA/ARMA → ARIMA/SARIMA → estimation → residual validation → model selection → forecasting.**

The API pages then show how each mathematical object becomes executable Python, while the worked examples demonstrate the complete workflow on data.

## Source note

This page is a StochX documentation adaptation of the supplied USTHB course material. The mathematical terminology and sequence above follow the source document; the Python mapping is to the current StochX public API.