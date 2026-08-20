# ACF, PACF and the correlogram

## What these tools answer

The autocorrelation function (ACF) describes linear dependence between $y_t$ and $y_{t-k}$.

$$\rho_k=\frac{\operatorname{Cov}(y_t,y_{t-k})}{\sqrt{\operatorname{Var}(y_t)\operatorname{Var}(y_{t-k})}}.$$

The partial autocorrelation function (PACF) measures the remaining lag-$k$ association after accounting for the intermediate lags.

StochX exposes both directly:

```python
from stochx.timeseries import acf, pacf, correlogram

acf_result = acf(y, nlags=24)
pacf_result = pacf(y, nlags=24)
corr = correlogram(y, nlags=24)

print(corr.table())
```

## Why the correlogram is central

The `CorrelogramResult` combines the views an EViews user normally inspects together: lag, AC, PAC, Ljung–Box Q-statistic, p-value, degrees of freedom and confidence bands.

The report is intentionally table-oriented so it can be read in notebooks, tests and automated reports.

## Interpretation patterns

A practical identification guide is:

| Pattern | Typical indication |
|---|---|
| ACF cuts off, PACF tails | MA-type behavior |
| PACF cuts off, ACF tails | AR-type behavior |
| Both tail off | ARMA-type behavior |
| Strong seasonal spikes | seasonal structure may be present |
| Slow decay at low lags | possible non-stationarity |

These are heuristics, not proofs. The selected model must still be estimated and validated.

## Ljung–Box and Box–Pierce

The correlogram also provides residual serial-correlation diagnostics. For a residual series $e_t$, the Ljung–Box statistic aggregates autocorrelation evidence over a set of lags.

```python
from stochx.timeseries import ljung_box

result = ljung_box(residuals, lags=(1, 2, 4, 8))
print(result)
```

Use residual diagnostics after fitting rather than relying only on the original series' correlogram.

## Numerical conventions

The StochX correlogram uses explicit finite-sample counts and confidence-band conventions. These conventions are frozen by regression fixtures because tiny changes in lag caps, effective sample sizes or standard-error formulas can change downstream identification decisions.

## Practical workflow

1. Plot or inspect the raw series.
2. Make the series stationary if required.
3. Compute ACF/PACF.
4. Inspect the combined correlogram.
5. Propose a small set of candidate orders.
6. Estimate candidates.
7. Check residuals.

Continue to [Models](models.md) and [Box–Jenkins](box-jenkins.md).
