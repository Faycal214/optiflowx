# API quick map

The detailed API reference is organized by public class, following a PyDTMC-style documentation pattern.

## Domain classes

| Domain | Reference |
|---|---|
| CMTD | [`MarkovChain`](api/markov_chain.md) |
| Poisson | [`PoissonProcess`](api/poisson_process.md) |
| NHPP | [`NonHomogeneousPoissonProcess`](api/poisson_process.md#nonhomogeneouspoissonprocess) |
| CMTC | [`ContinuousTimeMarkovChain`](api/continuous_time_markov_chain.md) |
| CTMC trajectories | [`CTMCPath`](api/continuous_time_markov_chain.md#ctmcpath) |
| Birth-death | [`BirthDeathProcess`](api/birth_death_process.md) |
| Probability space | [`FiniteProbabilitySpace`](api/probability_space.md) |
| Random variable | [`RandomVariable`](api/random_variable.md) |
| Partition | [`Partition`](api/partition.md) |
| Filtration | [`Filtration`](api/filtration.md) |
| Martingale | [`Martingale`](api/martingale.md) |
| Stopping time | [`StoppingTime`](api/stopping_time.md) |
| Stopped process | [`StoppedProcess`](api/stopped_process.md) |

## Standalone utility

[`empirical_state_frequencies`](api/analysis.md) is kept standalone because it analyzes arbitrary trajectories rather than belonging to one process class.

## Mathematical definitions

The API pages explain how to use the Python objects. Mathematical definitions, propositions, and course-faithful explanations remain in the separate [Course material](course_material.md) section.

## Package architecture

See [Architecture](architecture.md) for the dependency and implementation structure.
