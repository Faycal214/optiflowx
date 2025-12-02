# OptiFlowX

[![PyPI version](https://img.shields.io/pypi/v/optiflowx)](https://pypi.org/project/optiflowx/)
[![Python versions](https://img.shields.io/pypi/pyversions/optiflowx.svg)](https://pypi.org/project/optiflowx/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/Faycal214/optiflowx/actions/workflows/test.yml/badge.svg)](https://github.com/Faycal214/optiflowx/actions)


**OptiFlowX** is an open-source framework for **hyperparameter optimization** using **combinatorial and metaheuristic algorithms**.
It enables systematic exploration of search spaces for machine learning models using advanced stochastic and hybrid optimization techniques.

---

## 🔍 Overview
Machine learning models rely on carefully tuned hyperparameters. Conventional methods
such as grid or simple random search become inefficient for discrete, mixed or
high-dimensional search spaces.

**OptiFlowX** is a modular framework for hyperparameter and configuration
optimization that brings together combinatorial and metaheuristic algorithms
so you can search complex spaces with a consistent, extensible API.

Key capabilities include:

- Flexible support for classification and regression tasks via a `task_type`
    parameter.
- First-class support for user-provided `custom_metric(y_true, y_pred)` callables
    (the framework will use them for cross-validation and accept them in
    parallel workers, with a `dill` fallback for non-pickleable callables).
- Unified scoring interface: built-in metrics (accuracy, f1, mse, rmse, mae,
    r2) are normalized so optimizers can always maximize a single numeric score.

---

## ⚙️ Key Features

- Unified interface for multiple optimization algorithms.
- Works with discrete, categorical, and continuous parameter spaces.
- Includes multiple strategies:
  - Genetic Algorithm (GA)
  - Particle Swarm Optimization (PSO)
  - Bayesian Optimization
  - Tree-structured Parzen Estimator (TPE)
  - Random Search
  - Simulated Annealing
- Scalable, parallel execution support.
- Compatible with any ML framework (scikit-learn, PyTorch, TensorFlow, etc.).

Additional highlights:

- Custom metric support: pass a callable `def my_metric(y_true, y_pred) -> float`
    and the engine will use it for evaluation (supports multiprocessing via
    a serialization fallback).
- Regression-ready: built-in regression metrics (`mse`, `rmse`, `mae`, `r2`)
    are provided and normalized so the optimizer maximizes performance.
- Consistent return format: optimizers return `(best_params, best_score)` so
    results are easy to consume programmatically.
---

## ⚗️ Installation

```bash
pip install optiflowx
```

or from source:

```bash
git clone https://github.com/Faycal214/optiflowx.git
cd optiflowx
pip install -e .
```

---

## 🧠 Example Usage

```python
from sklearn.datasets import make_classification, make_regression
from optiflowx.optimizers.genetic import GeneticOptimizer
from optiflowx.optimizers.pso import PSOOptimizer
from optiflowx.models.configs.random_forest_config import RandomForestConfig


# Example 1 — classification with a built-in metric
X, y = make_classification(n_samples=200, n_features=12, random_state=0)
cfg = RandomForestConfig()
search_space = cfg.build_search_space()
model_wrapper = cfg.get_wrapper(task_type="classification")

opt = PSOOptimizer(
    search_space=search_space,
    metric="accuracy",
    model_class=model_wrapper.model_class,
    X=X,
    y=y,
    n_particles=12,
)
best_params, best_score = opt.run(max_iters=5)
print("Classification result:", best_score, best_params)


# Example 2 — regression with built-in metric
Xr, yr = make_regression(n_samples=200, n_features=8, noise=0.1, random_state=1)
cfg = RandomForestConfig()
reg_wrapper = cfg.get_wrapper(task_type="regression")

optr = GeneticOptimizer(
    search_space=search_space,
    metric="mse",  # mse is negated internally so higher is better
    model_class=reg_wrapper.model_class,
    X=Xr,
    y=yr,
    population=20,
)
best_params_r, best_score_r = optr.run(max_iters=5)
print("Regression result:", best_score_r, best_params_r)


# Example 3 — custom metric callable
def my_custom_metric(y_true, y_pred):
    """User-defined metric: return any float (higher is better)."""
    # e.g., negative mean absolute error (so higher is better)
    from sklearn.metrics import mean_absolute_error

    return -float(mean_absolute_error(y_true, y_pred))

opt_custom = PSOOptimizer(
    search_space=search_space,
    metric="accuracy",
    custom_metric=my_custom_metric,
    model_class=model_wrapper.model_class,
    X=X,
    y=y,
    n_particles=8,
)
best_params_c, best_score_c = opt_custom.run(max_iters=4)
print("Custom metric result:", best_score_c, best_params_c)
```

---

## 🧪 Testing


```bash
pytest -v --maxfail=1 --disable-warnings
```

## 🧩 Optional dependency: `dill`

If you plan to pass non-top-level (non-pickleable) `custom_metric` callables
to optimizers while using multiprocessing, the project can optionally use
`dill` to serialize those callables. If `dill` is not installed, custom
metrics must be pickleable (e.g., top-level functions).

Install `dill` in your environment with:

```bash
pip install dill
```

The framework will automatically attempt to use `dill` when necessary; if it
is not present and a non-pickleable callable is provided, an error will be
raised with guidance to install `dill` or provide a pickleable metric.

---

## 📚 Citation

```
@software{optiflowx,
    author = {Faycal, Alikacem},
    title = {OptiFlowX: Combinatorial Hyperparameter Optimization Framework},
    year = {2025},
    url = {https://github.com/Faycal214/optiflowx}
}
```
