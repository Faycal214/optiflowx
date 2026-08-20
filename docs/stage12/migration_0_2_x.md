# Migrating from StochX 0.2.x

## Package identity

Keep using the canonical imports:

```python
import stochx
from stochx import stochastic, timeseries
```

No package-renaming step is required.

## Existing stochastic-process code

Existing `stochx.stochastic` objects remain the primary compatibility surface. Markov chains, Poisson processes, continuous-time Markov chains, birth-death processes, finite probability spaces, filtrations, martingales, stopping times, and stopped processes continue to be available.

## Time-series additions

The newer release line adds an explicit Box–Jenkins workflow around the existing time-series infrastructure:

```python
from stochx.timeseries import identify_box_jenkins

identification = identify_box_jenkins(series, d=1)
```

Identification, candidate estimation, residual validation, deterministic model selection, and forecasting are additive. Existing AR/MA/ARMA/ARIMA/SARIMA APIs do not need to be rewritten.

## State-space additions

For a scalar local-level model:

```python
from stochx.timeseries import local_level_filter

result = local_level_filter(series, process_variance=0.2, observation_variance=0.4)
```

For a general linear-Gaussian system:

```python
from stochx.timeseries import LinearStateSpace, kalman_filter
```

Stage 11 extends the filter with smoothing, forecasting, likelihood estimation, innovation diagnostics, adequacy checks, and an integrated workflow. These features compose around the Stage 10 filtering result rather than replacing it.

## Missing observations

State-space code should normally pass missing values through to the filter instead of deleting rows manually. Missing dimensions are tracked explicitly; fully missing time rows remain in the sequence.

## Release and versioning

The package version remains single-sourced. The release candidate version is selected later in the Stage 12 release process.

## Compatibility rule

When upgrading from `0.2.x`, existing code should continue to run unchanged unless it opts into a new API. Numerical regressions should be rerun for applications that depend on exact floating-point values or published benchmark outputs.
