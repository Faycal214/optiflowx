# ACF, PACF and the correlogram

## What these tools answer

The correlogram is a central EViews time-series view: it displays the autocorrelation and partial autocorrelation functions together with the Ljung–Box Q-statistic and its probability. EViews also allows the correlogram to be computed for the level, first difference, or higher difference of a series. citeturn104974search2turn193586search5

StochX exposes the same numerical components directly:

```python
from stochx.timeseries import acf, pacf, correlogram

acf_result = acf(y, nlags=24)
pacf_result = pacf(y, nlags=24)
corr = correlogram(y, nlags=24)

print(corr.table())
```

For a workfile, the active sample is applied first:

```python
wf.set_sample("2010M1 2020M12")
corr = wf.correlogram("Y", nlags=24)
corr_d1 = wf.correlogram("Y", nlags=24, d=1)
```

## Autocorrelation

EViews estimates the sample autocorrelation at lag (k) with one common overall sample mean:

[
	au_k =
rac{
sum_{t=k+1}^{T}
(Y_t-ar Y)(Y_{t-k}-ar Y)
}{
sum_{t=1}^{T}(Y_t-ar Y)^2
}.
]

EViews explicitly notes that this differs slightly from estimators using separate means for the shortened lagged sample. citeturn193586search3turn193586search5

StochX implements this EViews convention rather than using a generic correlation coefficient.

## Partial autocorrelation

EViews computes PAC recursively from the estimated autocorrelations using the Box–Jenkins / Durbin–Levinson construction. For lag (k),

[
phi_{kk}
=
rac{
	au_k-sum_{j=1}^{k-1}phi_{k-1,j}	au_{k-j}
}{
1-sum_{j=1}^{k-1}phi_{k-1,j}	au_j
},
]

followed by

[
phi_{k,j}
=
phi_{k-1,j}
-
phi_{kk}phi_{k-1,k-j}.
]

EViews describes this as a consistent approximation to the partial autocorrelation and notes that a direct regression can provide a more precise estimate. citeturn104974search6turn193586search6

## Confidence bands

EViews displays approximate two-standard-error limits:

[
pm rac{2}{sqrt{T}}.
]

These bands are approximate and are based on the effective observation count (T). The band convention is independent of the requested significance setting in the StochX correlogram API; the explicit `alpha` parameter is used for result-level decision interpretation rather than changing these EViews-style bands. citeturn104974search6

## Ljung–Box Q-statistic

For lag (k), EViews computes:

[
Q_{LB}
=
T(T+2)
sum_{j=1}^{k}
rac{	au_j^2}{T-j}.
]

For an ordinary series, the asymptotic reference distribution uses (k) degrees of freedom. For residuals from an ARIMA model, EViews adjusts the degrees of freedom by the number of estimated AR and MA terms. citeturn104974search6

StochX stores the adjusted degrees of freedom in `result.DF` and returns undefined probabilities when the adjusted degrees of freedom are non-positive.

## Display contract

The default `CorrelogramResult.table()` matches the EViews statistical table:

```text
Lag    AC    PAC    Q-Stat    Prob.
```

The adjusted degrees of freedom and confidence bands remain available from the result object and may be appended explicitly for auditing:

```python
corr.DF
corr.ac_lower
corr.ac_upper

corr.table(include_df=True, include_bands=True)
```

## Missing observations

StochX currently performs one common missing-value preprocessing pass for direct ACF/PACF/correlogram calculations: NaNs are removed before the numerical calculation and the resulting effective (T) is used consistently for the ACF, PACF, Q-statistic and confidence bands.

This is an explicit StochX convention. When exact EViews missing-data behavior needs to be certified for a particular data layout, it must be frozen against an actual EViews output fixture rather than inferred.

## Interpretation patterns

| Pattern | Typical indication |
|---|---|
| ACF cuts off, PACF tails | MA-type behavior |
| PACF cuts off, ACF tails | AR-type behavior |
| Both tail off | ARMA-type behavior |
| Strong seasonal spikes | seasonal structure may be present |
| Slow decay at low lags | possible non-stationarity |

These are identification heuristics, not proofs. The selected model must still be estimated and validated.

## Practical workflow

1. Plot or inspect the raw series.
2. Make the series stationary if required.
3. Compute ACF/PACF.
4. Inspect the combined correlogram.
5. Propose a small set of candidate orders.
6. Estimate candidates.
7. Check residuals.

Continue to [Models](models.md) and [Box–Jenkins](box-jenkins.md).
