# Stage 8.10 EViews correlogram reference fixtures

These fixtures define the deterministic parity schema for the StochX correlogram.

## Reference settings

- AC: EViews common-overall-mean convention.
- PAC: recursive Box-Jenkins / Durbin-Levinson construction.
- Q-Stat: Ljung-Box `n(n+2) * sum(rho_j^2 / (n-j))`.
- DF: `lag - model_df`.
- Prob.: chi-square survival probability for positive DF; `NaN` when DF <= 0.
- Missing values: remove NaNs once before calculation; reject infinities.
- Displayed lags: 1 through `nlags`.
- Confidence bands: approximate EViews two-standard-error bands, `+/- 2/sqrt(nobs)`.
- `alpha`: Q-statistic decision level; it does not change the EViews-style band width.

## Fixture status

The JSON file is intentionally marked `convention_derived_pending_eviews_export` because the repository does not currently contain a captured EViews Stage 8 output package. The expected arrays are hard-coded from the frozen conventions and deterministic raw series, so parity tests are independent of StochX's implementation. When real EViews captures become available, replace the expected arrays while preserving the schema and settings.

## Cases

- `ordinary`: ordinary series, `nlags=6`, `model_df=0`, `alpha=0.05`.
- `residual`: residual-style series, `nlags=6`, `model_df=2`, `alpha=0.05`.
