# API Reference

OptiFlowX exposes stochastic-process objects as explicit Python classes. The reference follows a PyDTMC-style layout: each class page describes the mathematical object, constructor, public properties, public methods, examples, and related course material.

## Markov chains

- [`MarkovChain`](markov_chain.md) — finite homogeneous discrete-time Markov chains.

## Poisson processes

- [`PoissonProcess`](poisson_process.md) — homogeneous Poisson process.
- [`NonHomogeneousPoissonProcess`](poisson_process.md#nonhomogeneouspoissonprocess) — intensity-driven Poisson process.

## Continuous-time Markov chains

- [`ContinuousTimeMarkovChain`](continuous_time_markov_chain.md) — finite homogeneous CTMCs.
- [`CTMCPath`](continuous_time_markov_chain.md#ctmcpath) — simulated CTMC trajectories.
- [`BirthDeathProcess`](birth_death_process.md) — birth-death specialization of a CTMC.

## Conditional expectation

- [`FiniteProbabilitySpace`](probability_space.md)
- [`RandomVariable`](random_variable.md)
- [`Partition`](partition.md)

## Martingales

- [`Filtration`](filtration.md)
- [`Martingale`](martingale.md)
- [`StoppingTime`](stopping_time.md)
- [`StoppedProcess`](stopped_process.md)

## Standalone analysis

- [`empirical_state_frequencies`](analysis.md) — generic trajectory-frequency analysis.

## Exceptions

- [`Exception hierarchy`](exceptions.md) — semantic validation and numerical exceptions.

## API conventions

Properties describe intrinsic information already attached to an object. Methods perform computations or operations that require arguments. Public state labels are preserved at the API boundary while numerical implementations may use integer indices internally.

The API reference is separate from the mathematical Course material. For definitions and proofs, use the corresponding course chapter first.
