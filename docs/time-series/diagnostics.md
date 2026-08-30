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

## EViews equation-level diagnostics

EViews exposes residual correlograms/Q-statistics, squared-residual correlograms, histogram/normality, serial-correlation LM, and heteroskedasticity tests from the estimated equation. EViews also provides separate stability diagnostics. 

StochX now attaches these to EquationResult:

    eq.diagnostics(lags=12)
    eq.residual_correlogram(lags=12)
    eq.squared_residual_correlogram(lags=12)
    eq.normality_test()
    eq.serial_correlation_lm(lags=4)
    eq.heteroskedasticity(test='BPG')
    eq.heteroskedasticity(test='Harvey')
    eq.heteroskedasticity(test='Glejser')
    eq.heteroskedasticity(test='ARCH', lags=12)
    eq.heteroskedasticity(test='White', cross_terms=False)

### Residual correlogram

EViews adjusts the Ljung-Box degrees of freedom for included ARMA terms. StochX uses the number of estimated ordinary and seasonal ARMA terms for the equation-level adjustment.

### Squared residual correlogram

EViews computes AC, PAC and Q-statistics from squared residuals as an ARCH diagnostic. StochX exposes this as a separate equation view.

### Histogram / normality

EViews reports residual descriptive statistics and the Jarque-Bera normality statistic with two degrees of freedom.

### Serial correlation LM

EViews exposes the Breusch-Godfrey LM test as equation view auto(order), reporting Obs*R-squared/LM and the auxiliary F test. EViews modifies the auxiliary regression when ARMA terms are included. The StochX standard implementation currently returns the standard Breusch-Godfrey auxiliary regression and retains the model degrees-of-freedom metadata; exact ARMA-modified numerical parity remains fixture-dependent.

### Heteroskedasticity

EViews hettest supports BPG, Harvey, Glejser, ARCH and White, with White cross-products optional. StochX exposes those same test families through EquationResult. 

### Stability

EViews recursive residuals, CUSUM and CUSUM of squares are OLS-only recursive stability views; Chow breakpoint and Chow forecast are separate stability procedures. StochX exposes these through stability_diagnostics and rejects recursive stability requests for ARMA-error equations.

### Parity boundary

Interface and documented test families are implemented. Exact numerical EViews parity still requires captured EViews fixtures for the ARMA-modified LM auxiliary regression, exact CUSUM/CUSUMSQ critical lines, Chow forecast LR details, and every heteroskedasticity auxiliary-regression option.