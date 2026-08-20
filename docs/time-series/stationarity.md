# Stationarity and unit-root tests

## Why stationarity matters

Many classical time-series results assume a stable stochastic structure. A weakly stationary series has time-invariant moments and autocovariances that depend on lag rather than calendar time.

In practice, a series with a changing level or unit root can make ordinary regression and ARMA interpretations misleading. StochX therefore treats stationarity as a decision step before model identification.

## Dickey–Fuller and ADF

The public API provides:

```python
from stochx.timeseries import adf, dickey_fuller, dickey_fuller_sequential

result = adf(y, regression="c", lags=1, autolag=None)
print(result.summary())
print(result.interpret())
```

The deterministic specification is important. Typical choices distinguish:

- no constant and no trend;
- constant only;
- constant and trend.

The null hypothesis is a unit root. The reported statistic must be compared with the correct critical values for the selected deterministic specification rather than with an ordinary Student-t cutoff.

## Sequential DF workflow

`dickey_fuller_sequential` implements the course-style sequential decision tree when the user needs a structured treatment of deterministic terms. The algorithm keeps the selected lag order common across the tested specifications and exposes the intermediate specification results.

## KPSS and Phillips–Perron

StochX also exposes `kpss_test` and `phillips_perron`. These are useful as complementary evidence because they use different null hypotheses or correction mechanisms.

Do not interpret one p-value in isolation. A robust workflow considers the series plot, the deterministic specification, the ADF result and complementary tests together.

## Difference after the decision

If the evidence supports differencing, apply it explicitly:

```python
from stochx.timeseries import difference

dy = difference(y, periods=1)
```

Then repeat the stationarity check on the transformed series.

## Common mistakes

- Using an ADF specification that does not match the visual/data-generating context.
- Choosing the lag order after looking at the final statistic without documenting the rule.
- Treating failure to reject a unit-root null as proof that the series is exactly integrated of order one.
- Differencing a stationary series without checking the resulting dynamics.

## Next step

Once the transformed series is plausibly stationary, inspect [ACF/PACF and the correlogram](correlation.md) to identify the dynamic orders.
