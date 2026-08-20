# Changelog

All notable changes to StochX are documented here.

## Unreleased — Stage 12 release hardening

### Added

- Completed Stage 8 correlogram and deterministic time-series contract work.
- Completed Stage 9 Box–Jenkins identification, candidate estimation, residual validation, deterministic selection, and forecasting workflows.
- Completed Stage 10 linear-Gaussian state-space filtering and Kalman-filter regression coverage.
- Completed Stage 11 state-space smoothing, forecasting, likelihood estimation, innovation diagnostics, adequacy testing, and end-to-end workflow integration.
- Added deterministic numerical fixtures, edge-case regression tests, and cross-stage compatibility gates across the frozen workflows.
- Expanded runnable examples and public time-series documentation to include the state-space workflows.

### Release hardening

- Added distribution-build and clean-wheel installation validation to CI.
- Added Stage 12 release-surface documentation and regression coverage.
- Kept the package version at `0.2.0` pending an explicit release-candidate decision.

### Compatibility

- Stage 8, Stage 9, Stage 10, and Stage 11 numerical and public API contracts remain frozen.

## [0.2.0] - 2026-08-15

### Added

- First release under the StochX project identity and `stochx` Python import namespace.
- Repository-wide migration of package imports, documentation, examples, tests, CI, release metadata, and citation metadata from the former project identity.
- Clean release verification covering the renamed package, full test suite, executable examples, strict documentation build, distribution build, and Twine metadata validation.

### Changed

- Promoted the package version from `0.1.0` to `0.2.0` for the first StochX release.
- Standardized the project, package, documentation, and release surfaces around `stochx` / StochX.

### Notes

- This is an early development release; public APIs may continue to evolve before the first stable `1.0.0` release.

## [0.1.0] - 2026-08-15

### Added

- Unified stochastic-process API covering DTMCs, Poisson processes, CTMCs, birth-death processes, finite probability spaces, conditional expectation, filtrations, martingales, stopping times, and stopped processes.
- Centralized stochastic-matrix, generator, probability-vector, state-label, and tolerance validation.
- CTMC transition probabilities through both matrix-exponential and uniformization routes.
- Explicit `CTMCPath` trajectory analysis with state lookup and occupation statistics.
- Dedicated PyDTMC-style class reference pages and runnable API examples.
- Automated documentation, API, docstring, and example coverage gates in CI.
- PyPI-ready package metadata and a tag-based trusted-publishing workflow.

### Changed

- Finalized the package identity around stochastic-process mathematics instead of the former optimization/HPO architecture.
- Standardized the public stochastic API and documentation conventions.
- Reworked the README around installation, public API, examples, testing, and release usage.

### Notes

- The project is in an early development stage and public APIs may continue to evolve before the first stable `1.0.0` release.
