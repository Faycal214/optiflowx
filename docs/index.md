# OptiFlowX

OptiFlowX is a lightweight library for hyperparameter optimization using combinatorial and population-based algorithms.
It provides ready-to-use implementations of Genetic Algorithms, Particle Swarm Optimization, Simulated Annealing, TPE/Optuna, Bayesian (scikit-optimize), and Random Search, plus model config wrappers and a SearchSpace abstraction.

## Key features

- Multiple optimizers with consistent APIs (GA, PSO, SA, Bayesian, TPE, Random Search).
- `SearchSpace` abstraction for continuous, discrete and categorical parameters.
- Model configuration registry with ready configs (RandomForest, SVC, XGBoost, MLP, etc.).
- Parallel evaluation support via `ParallelExecutor`.
- Simple programmatic API for experiments and integration into pipelines.
- Extensible: add new models or optimizers with minimal glue.

## Why it matters

Hyperparameter tuning is essential for ML performance but can be complex and time-consuming. OptiFlowX offers:
- Fast prototyping of optimization strategies.
- A simple, reproducible experiment loop.
- Flexibility to use algorithmic strategies that suit your search space and compute budget.

## Quick links

- Getting started: `docs/getting-started.md`
- Examples: `docs/examples.md`
- Algorithms explained: `docs/algorithms.md`
- API reference: `docs/api.md`
- Contributing: `docs/contributing.md`
