# Stage 10.3 deterministic state-space fixtures

These fixtures are independent numerical references for the Stage 10.2 linear-Gaussian state-space and Kalman-filter core.

## Reference hierarchy

1. `scalar_local_level.json`: hand/closed-form scalar Kalman recursion.
2. `scalar_dynamic.json`: independently evaluated scalar recursion with fixed matrices and five observations.
3. `multivariate_missing.json`: independently evaluated two-state/two-observation recursion with partial and fully missing observations.

The expected values in these files are not generated at test time by importing the StochX filtering implementation.

## Reproducibility

Every fixture records all model matrices, initial state/covariance, observations, and expected outputs. No fixture depends on ambient randomness or wall-clock state.

## Tolerances

Structural counts and observation masks are checked exactly. Floating-point state/covariance/innovation values use `rtol=1e-10`, `atol=1e-12` except the hand-calculated local-level fixture, which is checked at `1e-12` absolute tolerance. Log likelihood uses `rtol=1e-10`, `atol=1e-10`.

## Missing observations

`nobs` counts time rows. `effective_nobs` counts observed scalar measurements. `missing_observations` counts missing scalar measurements.

A fully missing row performs prediction only and contributes no likelihood increment.
