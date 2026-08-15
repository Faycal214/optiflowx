# Changelog

All notable changes to StochX are documented here.

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
