# Stochastic-process API

The public `stochx.stochastic` namespace contains explicit mathematical objects for stochastic-process coursework and applied probability.

## Markov chains

- `MarkovChain`
- `empirical_state_frequencies`

See [MarkovChain](markov_chain.md) for transitions, communication classes, recurrence/transience, hitting and return times, stationarity, absorption and simulation.

## Poisson processes

- `PoissonProcess`
- `NonHomogeneousPoissonProcess`

These objects cover counting-process probabilities and reproducible arrival-path simulation.

## Continuous time

- `ContinuousTimeMarkovChain`
- `CTMCPath`
- `BirthDeathProcess`

The API uses generator matrices, continuous-time transition probabilities and explicit path objects.

## Probability objects

- `FiniteProbabilitySpace`
- `RandomVariable`
- `Partition`

These form the basis for finite-space expectations and conditional expectation.

## Information and martingales

- `Filtration`
- `Martingale`
- `StoppingTime`
- `StoppedProcess`

These objects model the information structure and stopping operations used in the corresponding mathematical theory.

## Exceptions and validation

The public hierarchy includes `StochXError`, `ValidationError`, `ProbabilityValidationError`, `MatrixValidationError`, `GeneratorValidationError` and `NumericalError`.

The package prefers explicit failures over silent coercion. When a model violates its mathematical assumptions, the failure should identify the violated invariant as close to the API boundary as possible.

For conceptual explanations start with the [Stochastic Processes User Guide](../stochastic/index.md).
