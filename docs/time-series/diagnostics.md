# Diagnostics and model adequacy

A fitted model is useful only when its residuals behave in a way that is compatible with the assumptions used for inference and forecasting.

## Core residual questions

For a residual series $e_t$, ask:

1. Is the residual mean compatible with zero?
2. Is serial correlation left in the residuals?
3. Is the residual distribution reasonably compatible with the intended inference?
4. Is there evidence of changing variance or ARCH effects?
5. Are there redundant regressors or obvious misspecification clues?

## Public tests

StochX exposes:

```python
from stochx.timeseries import (
    durbin_watson_test,
    breusch_godfrey,
    ljung_box,
    jarque_bera,
    normality_ks,
    arch_test,
    breusch_pagan,
)
```

The exact test object records the statistic, p-value and interpretation where supported.

## Residual correlogram

```python
from stochx.timeseries import residual_correlogram, residual_diagnostics

corr = residual_correlogram(residuals, nlags=12)
diag = residual_diagnostics(residuals, lags=12, alpha=0.05)
print(corr.table())
print(diag.summary())
```

For Box–Jenkins candidates, the residual Ljung–Box rule is part of model eligibility. This means diagnostics feed model selection rather than appearing only as an afterthought.

## Breusch–Godfrey

For regression equations with possible serial correlation, use:

```python
bg = equation.serial_correlation(lags=2)
```

or the standalone `breusch_godfrey` helper where appropriate.

## Information criteria are not diagnostics

A low AIC does not prove that residuals are adequate. StochX therefore separates:

- **fit quality** — likelihood and information criteria;
- **serial adequacy** — residual correlation tests;
- **distributional checks** — normality diagnostics;
- **variance checks** — ARCH/Breusch–Pagan style tests.

This separation mirrors the actual statistical workflow and prevents a single score from becoming a substitute for model checking.

## Interpretation pattern

A practical report should say **what was tested, which statistic was obtained, what the null hypothesis means, and whether the evidence changes the modeling decision**. StochX's `interpret()` helpers are intended to support that narrative while leaving the raw result accessible for reproducibility.
