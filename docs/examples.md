# Worked Examples

The repository contains focused runnable examples for each stochastic-process area and the public time-series/state-space workflows.

| Example | Coverage |
|---|---|
| `examples/01_discrete_markov_chain.py` | Core `MarkovChain` construction and discrete-time operations |
| `examples/02_poisson_process.py` | Core `PoissonProcess` usage |
| `examples/02_poisson_complete.py` | Extended homogeneous and non-homogeneous Poisson operations |
| `examples/03_continuous_markov_chain.py` | Core `ContinuousTimeMarkovChain` usage |
| `examples/03_cmtc_complete.py` | Extended CTMC, jump-chain, holding-time, cost, and path operations |
| `examples/04_birth_death_process.py` | Birth-death construction and core formulas |
| `examples/05_conditional_expectation.py` | Core finite conditional-expectation usage |
| `examples/05_conditional_expectation_complete.py` | Extended probability-space, partition, random-variable, and conditional operations |
| `examples/06_martingale.py` | Core discrete-time martingale usage |
| `examples/07_api_operations.py` | Public classes and meaningful public operations across the stochastic API |
| `examples/08_eviews_time_series_workflow.py` | EViews-style time-series workfile and equation workflow |
| `examples/09_state_space_kalman.py` | Linear-Gaussian state-space filtering and missing-observation semantics |
| `examples/10_state_space_workflow.py` | End-to-end Stage 11 filtering, smoothing, diagnostics, adequacy, and forecasting |
| `examples/api_quickstart.py` | Minimal public API entry point |

The examples are executable documentation. CI runs every `examples/*.py` script and builds the documentation strictly so that runnable examples remain synchronized with the public API.

The canonical state-space workflows are documented in [Time Series](time_series.md), while the release-readiness contract is documented in [Stage 12.3](stage12/12.3_release_surface_closure.md).
