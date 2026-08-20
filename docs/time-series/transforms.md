# Transformations and decomposition

## Lags and differences

Lag operators are the foundation of ARMA-style modeling and are available in familiar notation:

```python
wf.eval("Y(-1)")
wf.eval("Y(-2)")
wf.eval("D(Y)")
```

Equivalent object-level operations are available through the `Workfile` and `TimeSeries` APIs.

A first difference is

$$\Delta y_t = y_t-y_{t-1}.$$

Higher-order differences apply the operator repeatedly. The important practical rule is to verify that differencing is needed rather than using it automatically: over-differencing can introduce unnecessary moving-average structure and distort interpretation.

## Log transformations

For positive economic or business series, log transformations are often useful:

```python
wf.eval("LOG(CPI)")
wf.eval("DLOG(CPI)")
```

`DLOG(Y)` represents the difference of the log series and is commonly interpreted as an approximate growth rate for small changes.

## Seasonal differences

Seasonal differencing removes a repeated seasonal level:

$$\Delta_s y_t = y_t-y_{t-s}.$$

StochX exposes `seasonal_difference` for explicit seasonal lag removal. The seasonal period should come from the data frequency and the analyst's domain knowledge, not from a blind default.

## Deterministic terms

Time-series models often distinguish stochastic dynamics from deterministic components such as an intercept or trend. StochX keeps that distinction explicit in stationarity tests and equation specifications.

The EViews-inspired `@TREND` term is available in supported equation workflows:

```python
eq = wf.ls("Y C X @TREND", name="EQ01")
```

## Smoothing

Smoothing can be useful for visualization and signal extraction before formal modeling:

```python
from stochx.timeseries import moving_average, exponential_smoothing, holt, holt_winters

ma = moving_average(y, window=12)
level = exponential_smoothing(y, alpha=0.3)
trend = holt(y, alpha=0.3, beta=0.1)
seasonal = holt_winters(y, seasonal_periods=12)
```

Smoothing is not the same thing as fitting an ARIMA model. A smooth representation can help interpretation, while the statistical model should still be selected and validated on the original modeling objective.

## Decomposition

`decompose` exposes level, trend, seasonal and residual components when the selected decomposition method supports them. Seasonal indices are also available for explicit seasonal adjustment workflows.

## Practical decision rule

Use transformations to answer a specific modeling question:

- unstable variance → consider a log transform;
- changing deterministic level → inspect differencing and trend terms;
- persistent seasonal level → inspect seasonal differences/indices;
- noisy visualization → use smoothing for inspection, not as a substitute for residual validation.

After transformations, return to [stationarity](stationarity.md) and [correlation](correlation.md) before identifying a dynamic model.
