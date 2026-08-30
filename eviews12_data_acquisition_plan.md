# EViews reference acquisition plan — Ubuntu

## Policy
Ubuntu is the execution environment for StochX. EViews is not a runtime dependency.

The purpose of this plan is to obtain legitimate reference data and output from public EViews documentation/examples or from a licensed EViews installation elsewhere.

## Priority 1 — Uroot
Obtain the actual Uroot example data:
- CS
- GDP
- 1948Q3–1988Q4

Then capture:
- OLS equation: CS C GDP CS(-1)
- coefficient table
- model statistics
- Breusch-Godfrey LM(2)

## Priority 2 — Denmark Johansen
Use the public Denmark data source or the exact EViews example workfile when available.

Required variables:
- LRM
- LRY
- IBO
- IDE

EViews reference setup:
- 1974Q3–1987Q3
- lag interval 1 1
- restricted constant
- seasonal exogenous terms D1 D2 D3 where specified by the reference

## Priority 3 — ARIMA
Obtain the underlying series/regressors for the documented automatic-ARIMA example and capture:
- transformation
- differencing
- candidate count
- selected order
- coefficients
- information criteria
- forecasts
- forecast standard errors
- confidence limits

## Ubuntu local layout
validation_data/
  raw/
  prepared/
  eviews/
  reports/

None of these files are package runtime dependencies.

## Rule
Never replace a missing EViews dataset with simulated random values. A missing source is a missing validation case.
