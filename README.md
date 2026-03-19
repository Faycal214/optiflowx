# OptiFlowX

[![PyPI version](https://img.shields.io/pypi/v/optiflowx)](https://pypi.org/project/optiflowx/)
[![Python versions](https://img.shields.io/pypi/pyversions/optiflowx.svg)](https://pypi.org/project/optiflowx/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/Faycal214/optiflowx/actions/workflows/test.yml/badge.svg)](https://github.com/Faycal214/optiflowx/actions)

OptiFlowX is a modular hyperparameter optimization framework that unifies **metaheuristics and probabilistic optimization methods** under a single, consistent API.

Unlike most libraries that focus on a single optimization paradigm (e.g., Bayesian optimization), OptiFlowX enables **experimentation across multiple optimizer families**, making it ideal for research, benchmarking, and hybrid optimization strategies.

## Key capabilities:

- Optimize models for classification and regression tasks
- Support for scikit-learn models and user-supplied custom models
- Built-in metrics and user-provided custom metric callables
- Built-in search-space definitions and fully custom search spaces
- Parallel candidate evaluation with optional `dill` fallback for non-pickleable callables

Table of contents

- [Why OptiFlowX](#why-optiflowx)
- [Use Cases](#use-cases)
- [Core Strength](#core-strength)
- [Key features](#key-features)
- [Algorithms included](#algorithms-included)
- [Quickstart](#quickstart)
- [Examples](#examples)
- [Search space](#search-space)
- [Parallelism & multiprocessing notes](#parallelism--multiprocessing-notes)
- [API reference (quick)](#api-reference-quick)
- [Development & testing](#development--testing)
- [Contributing](#contributing)
- [License](#license)
- [Contact & citation](#contact--citation)

## Why OptiFlowX

Most hyperparameter optimization libraries specialize in a single approach:

- Optuna → Bayesian / TPE optimization  
- scikit-optimize → lightweight Bayesian optimization  
- DEAP → evolutionary algorithms  

OptiFlowX is designed differently:

- Combines **metaheuristics (PSO, GA, ACO, GWO, SA)** and **probabilistic methods (TPE, Bayesian)**  
- Provides a **unified API** across all optimizers  
- Enables **easy comparison and experimentation between optimization strategies**  

This makes it particularly suitable for:
- research and benchmarking  
- testing multiple optimization strategies on the same problem  
- building hybrid optimization workflows

## Use Cases

- Hyperparameter tuning for ML models  
- Comparing optimization algorithms  
- Research in metaheuristics and hybrid optimization  
- Building custom AutoML pipelines  

## Core Strength

The main strength of OptiFlowX is its ability to **treat different optimization paradigms as interchangeable components**.

This allows you to:
- switch optimizers with minimal code changes  
- compare their performance on the same search space  
- build hybrid optimization strategies  

## Key features

- Unified optimizer interface (PSO, GA, ACO, GWO, SA, TPE, Bayesian, Random Search)
- Classification and regression support
- Plug-and-play model configs (`optiflowx.models.configs.*`) and `ModelWrapper` for CV and final fitting
- Built-in metric helpers and a `get_metric()` abstraction that normalizes regression metrics for maximization
- Flexible `SearchSpace` supporting continuous, discrete, and categorical parameters with sampling and grid generation
- Parallel evaluation via `ParallelExecutor` with pickle/dill serialization fallbacks

## Algorithms included

| Optimizer | Type | Best for |
|----------|------|----------|
| PSO | Metaheuristic | Continuous optimization |
| GA | Evolutionary | Complex/discrete spaces |
| SA | Metaheuristic | Escaping local minima |
| ACO | Swarm | Combinatorial problems |
| GWO | Swarm | Exploration-heavy search |
| TPE | Probabilistic | Efficient search |
| Bayesian | Probabilistic | Sample-efficient tuning |
| Random | Baseline | Quick exploration |

- Metaheuristics / Combinatorial:
  - Particle Swarm Optimization (PSO)
  - Genetic Algorithm (GA)
  - Simulated Annealing (SA)
  - Ant Colony Optimization (ACO)
  - Grey Wolf Optimizer (GWO)
- Probabilistic / Bayesian & related:
  - Tree-Structured Parzen Estimator (TPE)
  - Bayesian Optimization
  - Random Search

All algorithms are implemented under `optiflowx.optimizers.*`. Most optimizers expose a `run(max_iters=...)` method that returns `(best_params, best_score)`.

- `best_params`: best configuration found  
- `best_score`: corresponding score  

## Quickstart

Install (recommended inside a virtual environment):

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
# Optional extras
pip install dill xgboost
```

Minimal example (runs in a few seconds):

```python
from sklearn.datasets import make_classification
from optiflowx.models.configs import RandomForestConfig
from optiflowx.optimizers import PSOOptimizer

X, y = make_classification(n_samples=200, n_features=12, random_state=0)
cfg = RandomForestConfig()
wrapper = cfg.get_wrapper(task_type="classification")

opt = PSOOptimizer(
  search_space=cfg.build_search_space(),
  metric="accuracy",
  model_class=wrapper.model_class,
  X=X, y=y,
  n_particles=12,
)
best_params, best_score = opt.run(max_iters=10)
print(best_score, best_params)
```

## Examples

The `examples/` directory contains runnable scripts covering common combinations:

- classification with sklearn models and sklearn/custom metrics
- regression with sklearn models and sklearn/custom metrics
- examples that use `CustomModelConfig` for user-defined model wrappers

Run one of the example scripts directly:

```bash
python examples/classification/classification_sklearn_model_sklearn_metric.py
```

## Search space

Use built-in configs for quick starts (e.g., `RandomForestConfig().build_search_space()`), or create custom spaces with `optiflowx.core.search_space.SearchSpace`:

```python
from optiflowx.core import SearchSpace

s = SearchSpace()
s.add("n_estimators", "discrete", [10, 50, 100, 200])
s.add("learning_rate", "continuous", [1e-3, 0.3], log=True)
s.add("criterion", "categorical", ["gini", "entropy"])
```

## Parallelism & multiprocessing notes

`ParallelExecutor` uses `multiprocessing.Pool` to evaluate candidates concurrently. If you pass non-pickleable callables (e.g., nested functions or closures) as custom metrics, the executor will:

1. Try to serialize with `pickle`.
2. If `pickle` fails and `dill` is installed, it will serialize using `dill`.
3. If serialization is not possible and multiple workers are requested, the executor raises an error. If only one worker is used it will fall back to sequential evaluation.

If you plan to pass nested custom metrics and want parallel execution, install `dill`:

```bash
pip install dill
```

## API reference (quick)

High-level building blocks (see docstrings for full signatures):

- `optiflowx.core.SearchSpace` — define and sample hyperparameter spaces
- `optiflowx.core.ModelWrapper` — cross-val evaluation and final fitting
- `optiflowx.core.get_metric` — normalized metric callables (negates regression errors so optimizers maximize)
- `optiflowx.core.ParallelExecutor` — parallel candidate evaluation
- `optiflowx.optimizers.*` — concrete optimizers (e.g., `PSOOptimizer`, `GeneticOptimizer`)
- `optiflowx.models.configs.*` — model configs exposing `build_search_space()` and `get_wrapper()`

Typical high-level flow:

1. Select model config from `optiflowx.models.registry.MODEL_REGISTRY`.
2. Build its `SearchSpace` and `ModelWrapper`.
3. Initialize an optimizer with the space, metric, `model_class`, and data.
4. Run `optimizer.run(max_iters=...)` or use `optiflowx.core.OptimizationEngine` / `MLPipeline` to orchestrate runs.

## Development & testing

Run tests locally (recommended in a virtualenv):

```bash
pip install -r requirements-test.txt
pytest -q
```

Developer tools (optional):

```bash
pip install black ruff mypy pytest pytest-cov
ruff check .
black .
mypy optiflowx
```

## Contributing

Contributions welcome. Please follow these steps:

1. Open an issue describing the feature or bug.
2. Create a topic branch in your fork.
3. Add tests for any new behavior.
4. Submit a PR with a clear description and changelog entry.

If you plan to add new optimizers or model configs, aim for:

- clear docstrings and examples under `examples/`;
- unit tests in `tests/` exercising the integration (optimizer + wrapper + executor);
- lightweight, focused commits.

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.

## Contact & citation

If you use OptiFlowX in research or production, please cite:

```bibtex
@software{optiflowx,
    author = {Faycal, Alikacem},
    title = {OptiFlowX: Combinatorial Hyperparameter Optimization Framework},
    year = {2025},
    url = {https://github.com/Faycal214/optiflowx}
}
```

Contact:

- Author: Alikacem Faycal
- Email: faycal213.dz@gmail.com

Acknowledgements

This project draws on many open-source tools and libraries including scikit-learn, Optuna, scikit-optimize, and others. See `pyproject.toml` for declared dependencies.

----

If you'd like, I can also:

- run the test suite in this environment and report results,
- produce a condensed one-page quick reference for the most common API calls,
- add badges for coverage/Docs/CodeQL if you want to include them in CI.


