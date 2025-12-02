# OptiFlowX

[![PyPI version](https://img.shields.io/pypi/v/optiflowx)](https://pypi.org/project/optiflowx/)
[![Python versions](https://img.shields.io/pypi/pyversions/optiflowx.svg)](https://pypi.org/project/optiflowx/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/Faycal214/optiflowx/actions/workflows/test.yml/badge.svg)](https://github.com/Faycal214/optiflowx/actions)

**OptiFlowX** is a production-oriented, modular framework for hyperparameter and configuration optimization.  
It unifies metaheuristic combinatorial algorithms, probabilistic optimizers, and search-space abstractions into a single, consistent API. Designed for both research and production, OptiFlowX supports:

- classification and regression,
- sklearn models and user-supplied custom models,
- sklearn metrics and user-supplied custom metrics,
- built-in search spaces and fully custom search spaces,
- efficient parallel evaluation (multiprocessing) with optional `dill` fallback for non-pickleable callables.

---

## Table of contents

- [Why OptiFlowX](#why-optiflowx)
- [Key features](#key-features)
- [Algorithms included](#algorithms-included)
- [Installation](#installation)
- [Quickstart](#Quickstart)
- [Classification examples](#classification-examples)
  - [sklearn model + sklearn metric](#sklearn-model+sklearn-metric)
  - [sklearn model + custom metric](#sklearn-model+custom-metric)
  - [sklearn model + sklearn metric](#sklearn-model+sklearn-metric)
  - [sklearn model + sklearn metric](#sklearn-model+sklearn-metric)
- [Regression examples](#regression-examples)
  - [sklearn model + sklearn metric](#sklearn-model+sklearn-metric)
  - [sklearn model + custom metric](#sklearn-model+custom-metric)
  - [sklearn model + sklearn metric](#sklearn-model+sklearn-metric)
  - [sklearn model + sklearn metric](#sklearn-model+sklearn-metric)
- [Search space: built-in vs custom (two examples)](#search-space-built-in-vs-custom-two-examples)
- [Parallelism & multiprocessing notes](#parallelism--multiprocessing-notes)
- [API reference (quick)](#api-reference-quick)
- [Testing & development](#testing--development)
- [Contact & citation](#contact--citation)

---

## Why OptiFlowX

- One consistent API to run many optimizers on any model and any metric.  
- Production-minded: packaging, examples, tests, docs (mkdocs), CI, and PyPI publishing in the repository.  
- Designed to be extensible — add new optimizers, new model wrappers, or new search-space primitives.

---

## Key features

- Unified interface across algorithms (PSO, GA, ACO, GWO, SA, TPE, Bayesian, Random Search).
- Support for both classification and regression tasks.
- Support for sklearn models (via wrappers) and user-provided custom models (custom wrapper interface).
- Support for built-in metrics (accuracy, f1, mse, rmse, mae, r2) and custom metric callables.
- Built-in search-space definitions for common models plus a flexible `SearchSpace` to build custom spaces programmatically.
- Parallel candidate evaluation (multiprocessing) with optional `dill` fallback for serializing complex callables.
- Examples split by task and by combinations of model/metric type (easy to run locally, in CI, or in Colab).

---

## Algorithms included

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

All algorithms are implemented as `Optimizer` subclasses under `optiflowx.optimizers.*` and share the same initialization parameters (`search_space`, `metric`/`custom_metric`, `model_class`, `X`, `y`, and algorithm-specific args).

---

## Installation

Recommended: use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # macOS / Linux
# or: .\venv\Scripts\activate  # Windows PowerShell

# Install latest published release
pip install optiflowx

# OR install directly from GitHub (editable)
pip install -e git+https://github.com/Faycal214/optiflowx.git#egg=optiflowx

# Optional: install recommended extras
pip install dill         # for serializing non-pickleable metrics in multiprocessing
pip install xgboost      # if you will use XGBoost configs/wrappers
```

## Quickstart

```python
from sklearn.datasets import make_classification
from optiflowx.models.configs.random_forest_config import RandomForestConfig
from optiflowx.optimizers.pso import PSOOptimizer

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

## Classification examples

### Sklearn model + sklearn metric

```python
"""
Run:
    python examples/classification/classification_sklearn_model_sklearn_metric.py
"""
from sklearn.datasets import make_classification
from optiflowx.models.configs.random_forest_config import RandomForestConfig
from optiflowx.optimizers.pso import PSOOptimizer

def main():
    X, y = make_classification(n_samples=200, n_features=8, random_state=0)
    cfg = RandomForestConfig()
    wrapper = cfg.get_wrapper(task_type="classification")

    opt = PSOOptimizer(
        search_space=cfg.build_search_space(),
        metric="accuracy",
        model_class=wrapper.model_class,
        X=X, y=y,
        n_particles=8,
    )
    best_params, best_score = opt.run(max_iters=3)
    print("sklearn model + sklearn metric ->", best_score, best_params)

if __name__ == "__main__":
    main()
```

### Sklearn model + custom metric

```python
"""
Run:
    python examples/classification/classification_sklearn_model_custom_metric.py
"""
from sklearn.datasets import make_classification
from optiflowx.models.configs.random_forest_config import RandomForestConfig
from optiflowx.optimizers.genetic import GeneticOptimizer

def my_accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())

def main():
    X, y = make_classification(n_samples=200, n_features=8, random_state=1)
    cfg = RandomForestConfig()
    wrapper = cfg.get_wrapper(task_type="classification")

    opt = GeneticOptimizer(
        search_space=cfg.build_search_space(),
        metric="accuracy",
        custom_metric=my_accuracy,
        model_class=wrapper.model_class,
        X=X, y=y,
        population=12,
    )
    best_params, best_score = opt.run(max_iters=3)
    print("sklearn model + custom metric ->", best_score, best_params)

if __name__ == "__main__":
    main()
```

### Custom model + sklearn metric

```python
"""
Classification: custom model + sklearn metric.

Run:
    python examples/classification/classification_custom_model_sklearn_metric.py
"""
from sklearn.datasets import make_classification
from optiflowx.models.configs.custom_model_config import CustomModelConfig
from optiflowx.optimizers.pso import PSOOptimizer

def main():
    X, y = make_classification(n_samples=150, n_features=6, random_state=2)
    cfg = CustomModelConfig()
    wrapper = cfg.get_wrapper()

    opt = PSOOptimizer(
        search_space=cfg.build_search_space(),
        metric="accuracy",
        model_class=wrapper.model_class,
        X=X, y=y,
        n_particles=6,
    )
    best_params, best_score = opt.run(max_iters=3)
    print("custom model + sklearn metric ->", best_score, best_params)

if __name__ == "__main__":
    main()
```

### Custom model + custom metric

```python
"""
Classification: custom model + custom metric.

Run:
    python examples/classification/classification_custom_model_custom_metric.py
"""
from sklearn.datasets import make_classification
from optiflowx.models.configs.custom_model_config import CustomModelConfig
from optiflowx.optimizers.genetic import GeneticOptimizer

def simple_score(y_true, y_pred):
    return float((y_true == y_pred).mean())

def main():
    X, y = make_classification(n_samples=150, n_features=6, random_state=3)
    cfg = CustomModelConfig()
    wrapper = cfg.get_wrapper()

    opt = GeneticOptimizer(
        search_space=cfg.build_search_space(),
        metric="accuracy",
        custom_metric=simple_score,
        model_class=wrapper.model_class,
        X=X, y=y,
        population=8,
    )
    best_params, best_score = opt.run(max_iters=3)
    print("custom model + custom metric ->", best_score, best_params)

if __name__ == "__main__":
    main()
```

## Regression examples

### Sklearn model + sklearn metric

```python
"""
Regression: sklearn model + sklearn metric (use normalized mse via get_metric).

Run:
    python examples/regression/regression_sklearn_model_sklearn_metric.py
"""
from sklearn.datasets import make_regression
from optiflowx.models.configs.random_forest_config import RandomForestConfig
from optiflowx.core.metrics import get_metric
from optiflowx.optimizers.pso import PSOOptimizer

def main():
    X, y = make_regression(n_samples=200, n_features=8, noise=0.2, random_state=0)
    cfg = RandomForestConfig()
    wrapper = cfg.get_wrapper(task_type="regression")

    opt = PSOOptimizer(
        search_space=cfg.build_search_space(),
        metric="mse",
        custom_metric=get_metric("mse"),
        model_class=wrapper.model_class,
        X=X, y=y,
        n_particles=8,
    )
    best_params, best_score = opt.run(max_iters=3)
    print("regression + sklearn metric (mse) ->", best_score, best_params)

if __name__ == "__main__":
    main()
```

### Sklearn model + custom metric

```python
"""
Regression: sklearn model + custom metric (neg RMSE).

Run:
    python examples/regression/regression_sklearn_model_custom_metric.py
"""
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from optiflowx.models.configs.random_forest_config import RandomForestConfig
from optiflowx.optimizers.genetic import GeneticOptimizer

def neg_rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return -(mse ** 0.5)

def main():
    X, y = make_regression(n_samples=200, n_features=8, noise=0.2, random_state=1)
    cfg = RandomForestConfig()
    wrapper = cfg.get_wrapper(task_type="regression")

    opt = GeneticOptimizer(
        search_space=cfg.build_search_space(),
        metric="mse",
        custom_metric=neg_rmse,
        model_class=wrapper.model_class,
        X=X, y=y,
        population=12,
    )
    best_params, best_score = opt.run(max_iters=3)
    print("regression + custom metric ->", best_score, best_params)

if __name__ == "__main__":
    main()
```

### Custom model + sklearn metric

```python
"""
Regression: custom model + sklearn metric.

Run:
    python examples/regression/regression_custom_model_sklearn_metric.py
"""
from sklearn.datasets import make_regression
from optiflowx.models.configs.custom_model_config import CustomModelConfig
from optiflowx.core.metrics import get_metric
from optiflowx.optimizers.pso import PSOOptimizer

def main():
    X, y = make_regression(n_samples=150, n_features=6, noise=0.1, random_state=2)
    cfg = CustomModelConfig()
    wrapper = cfg.get_wrapper()

    opt = PSOOptimizer(
        search_space=cfg.build_search_space(),
        metric="mse",
        custom_metric=get_metric("mse"),
        model_class=wrapper.model_class,
        X=X, y=y,
        n_particles=6,
    )
    best_params, best_score = opt.run(max_iters=3)
    print("custom regression model + sklearn metric ->", best_score, best_params)

if __name__ == "__main__":
    main()
```

### Custom model + custom metric

```python
"""
Regression: custom model + custom metric (neg MAE).

Run:
    python examples/regression/regression_custom_model_custom_metric.py
"""
from sklearn.datasets import make_regression
from optiflowx.models.configs.custom_model_config import CustomModelConfig
from optiflowx.optimizers.genetic import GeneticOptimizer
from sklearn.metrics import mean_absolute_error

def neg_mae(y_true, y_pred):
    return -float(mean_absolute_error(y_true, y_pred))

def main():
    X, y = make_regression(n_samples=150, n_features=6, noise=0.1, random_state=3)
    cfg = CustomModelConfig()
    wrapper = cfg.get_wrapper()

    opt = GeneticOptimizer(
        search_space=cfg.build_search_space(),
        metric="mse",
        custom_metric=neg_mae,
        model_class=wrapper.model_class,
        X=X, y=y,
        population=8,
    )
    best_params, best_score = opt.run(max_iters=3)
    print("custom regression model + custom metric ->", best_score, best_params)

if __name__ == "__main__":
    main()
```
## Search space: built-in vs custom

### Built-in search space (recommended for quick start)

```python
from optiflowx.models.configs.random_forest_config import RandomForestConfig

cfg = RandomForestConfig()
space = cfg.build_search_space()   # returns an optiflowx.core.search_space.SearchSpace instance
```

### Custom search space (full control)

```python
from optiflowx.core.search_space import SearchSpace

s = SearchSpace()
# discrete choices
s.add("n_estimators", "discrete", [10, 50, 100, 200])
# continuous with log-scale option
s.add("learning_rate", "continuous", [1e-3, 0.3], log=True)
# categorical
s.add("criterion", "categorical", ["gini", "entropy"])
# integer-like discrete ranges can be provided as lists as well
s.add("max_depth", "discrete", [None, 3, 5, 10, 20])
```
- Use built-in configs for models to speed-up onboarding.
- Use custom search space for fine-grained experiments or when supporting exotic parameters.

## Parallelism & multiprocessing notes

- The library evaluates candidates in parallel where the optimizer design permits (PSO, GA, ACO, etc.). Parallel evaluation uses Python `multiprocessing.Pool` with a `num_workers` parameter derived from `os.cpu_count()` by default.
- When you pass non-pickleable custom metric callables (e.g., nested functions, lambda closures), the parallel executor attempts to serialize them with `pickle`. If `pickle` fails and `dill` is installed, `dill` will be used automatically.
  - If you plan to pass nested callables and expect parallel execution, install `dill`:
    
    ```bash
    pip install dill
    ```
- You can control parallel workers from code (if supported by optimizer) or by passing a `num_workers` argument to the internal `ParallelExecutor` / optimizer where applicable.
- For CI or fast tests, the examples support environment flags:
   - `EXAMPLES_FAST_MODE=1` to bypass heavy runs and return quick trivial results.
   - `EXAMPLES_MAX_ITERS=3` (or set integer) to limit iterations used by example scripts.
 
## API reference — quick

These are the most used entry points; see docstrings for detailed signatures.

- `optiflowx.core.search_space.SearchSpace` — build and sample parameter spaces.
- `optiflowx.core.model_wrapper.ModelWrapper` — uniform wrapper for fit/predict and CV evaluation.
- `optiflowx.core.metrics.get_metric(name)` — returns a normalized callable metric (useful for regression error metrics that need to be negated).
- `optiflowx.optimizers.*` — concrete optimizers:
  - `PSOOptimizer`, `GeneticOptimizer`, `RandomSearchOptimizer`, `SimulatedAnnealingOptimizer`, `AntColonyOptimizer`, `GreyWolfOptimizer`, `TPEOptimizer`, `BayesianOptimizer`.
- `optiflowx.models.configs.*` — model-specific configurations with `build_search_space()` and `get_wrapper()`:
  - `RandomForestConfig`, `SVCConfig`, `MLPConfig`, `DecisionTreeConfig`, `KNNConfig`, `XGBoostConfig`, `CustomModelConfig`, etc.

Each optimizer's `run(max_iters=...)` returns `(best_params, best_score)`.

## Testing & development

Local quick checks:

```bash
# run unit tests
pytest -v --maxfail=1 --disable-warnings

# run examples (one of them)
python examples/classification/classification_sklearn_model_sklearn_metric.py
```

Recommended developer tooling (use a virtualenv):

```bash
pip install -e .[dev]    # if you have dev extras in pyproject
# or individually
pip install pytest pytest-cov black ruff mypy build twine mkdocs mkdocs-material
```

Coverage & linting:

```bash
pytest --cov=optiflowx
ruff check .
black --check .
mypy optiflowx
```

## Contact & citation

If you use OptiFlowX in research or production, please cite:

```bash
@software{optiflowx,
    author = {Faycal, Alikacem},
    title = {OptiFlowX: Combinatorial Hyperparameter Optimization Framework},
    year = {2025},
    url = {https://github.com/Faycal214/optiflowx}
}
```
