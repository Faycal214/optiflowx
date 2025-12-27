"""Run all optimizers for regression scenarios.

This file contains the regression examples previously bundled in
`run_all_optimizers.py`. It runs each optimizer on a small synthetic
regression dataset with both sklearn error-based metrics (negated via
`get_metric`) and custom error metrics.
"""
import os
import sys
import logging
import traceback

# Ensure project root is importable when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sklearn.datasets import make_regression
import os
from optiflowx.models.configs import RandomForestConfig, CustomModelConfig
from optiflowx.core import get_metric
from optiflowx.utils.sanitize import sanitize_params

from optiflowx.optimizers import (
    PSOOptimizer,
    GeneticOptimizer,
    BayesianOptimizer,
    TPEOptimizer,
    RandomSearchOptimizer,
    SimulatedAnnealingOptimizer,
    GreyWolfOptimizer,
    AntColonyOptimizer,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s")

OPTIMIZERS = [
    ("pso", PSOOptimizer, {"n_particles": 8}),
    ("genetic", GeneticOptimizer, {"population": 12}),
    ("bayesian", BayesianOptimizer, {"n_initial_points": 5}),
    ("tpe", TPEOptimizer, {"population_size": 6}),
    ("random_search", RandomSearchOptimizer, {"n_samples": 20}),
    ("simulated_annealing", SimulatedAnnealingOptimizer, {"population_size": 8}),
    ("grey_wolf", GreyWolfOptimizer, {"population_size": 8}),
    ("ant_colony", AntColonyOptimizer, {"colony_size": 8}),
]

# Fast mode for CI/tests: substitute a trivial optimizer that returns quickly.
if os.environ.get("EXAMPLES_FAST_MODE"):
    class FastOptimizer:
        def __init__(self, search_space=None, metric=None, model_class=None, X=None, y=None, **kwargs):
            self.search_space = search_space

        def run(self, max_iters=1):
            params = self.search_space.sample() if self.search_space is not None else {}
            return params, 0.0

    OPTIMIZERS = [("fast", FastOptimizer, {})]


def run_optimizer(opt_name, opt_cls, opt_kwargs, search_space, wrapper, X, y, metric="mse", custom_metric=None, task_type="regression"):
    try:
        print(f"\n--- Running {opt_name} | task={task_type} | custom_metric={'yes' if custom_metric else 'no'} | custom_model={'CustomModel' if wrapper.model_class.__name__ == 'CustomModel' else 'RF'} ---")
        optimizer = opt_cls(
            search_space=search_space,
            metric=metric,
            custom_metric=custom_metric,
            model_class=wrapper.model_class,
            X=X,
            y=y,
            **opt_kwargs,
        )
        # Propagate wrapper for consistent evaluation
        try:
            optimizer.wrapper = wrapper
        except Exception:
            pass
        iterations = int(os.environ.get("EXAMPLES_MAX_ITERS", "3"))
        best_params, best_score = optimizer.run(max_iters=iterations)
        try:
            best_params = sanitize_params(best_params)
        except Exception:
            pass
        print(f"Result {opt_name}: score={best_score}, params={best_params}")
    except Exception:
        print(f"ERROR running {opt_name}:")
        traceback.print_exc()


def regression_scenarios():
    print("\n== Regression scenarios ==")
    Xr, yr = make_regression(n_samples=150, n_features=8, noise=0.2, random_state=1)
    rf_cfg = RandomForestConfig()
    rf_wrapper = rf_cfg.get_wrapper(task_type="regression")
    custom_cfg = CustomModelConfig()
    custom_wrapper = custom_cfg.get_wrapper()

    # scenario 1: regression + sklearn metric (use callable get_metric to ensure proper sign)
    mse_fn = get_metric("mse")
    for name, cls, kwargs in OPTIMIZERS:
        run_optimizer(name, cls, kwargs, rf_cfg.build_search_space(), rf_wrapper, Xr, yr, metric="mse", custom_metric=mse_fn, task_type="regression")

    # scenario 2: regression + custom metric (neg RMSE)
    def neg_rmse(y_true, y_pred):
        from sklearn.metrics import mean_squared_error

        return -float(mean_squared_error(y_true, y_pred) ** 0.5)

    for name, cls, kwargs in OPTIMIZERS:
        run_optimizer(name, cls, kwargs, rf_cfg.build_search_space(), rf_wrapper, Xr, yr, metric="mse", custom_metric=neg_rmse, task_type="regression")

    # scenario 3: custom regression model + sklearn metric (use mse_fn)
    for name, cls, kwargs in OPTIMIZERS:
        run_optimizer(name, cls, kwargs, custom_cfg.build_search_space(), custom_wrapper, Xr, yr, metric="mse", custom_metric=mse_fn, task_type="regression")

    # scenario 4: custom regression model + custom metric (neg_mae)
    def neg_mae(y_true, y_pred):
        from sklearn.metrics import mean_absolute_error

        return -float(mean_absolute_error(y_true, y_pred))

    for name, cls, kwargs in OPTIMIZERS:
        run_optimizer(name, cls, kwargs, custom_cfg.build_search_space(), custom_wrapper, Xr, yr, metric="mse", custom_metric=neg_mae, task_type="regression")


if __name__ == "__main__":
    regression_scenarios()
"""Regression example scenarios for OptiFlowX.

Contains four examples (in one file):
- regression + sklearn metric
- regression + custom metric
- custom regression model + sklearn metric
- custom regression model + custom metric

These examples use small datasets and short runs for quick local testing.
"""
import os
import sys

# Ensure project root is on sys.path when running examples directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sklearn.datasets import make_regression
from optiflowx.models.configs import RandomForestConfig, CustomModelConfig
from optiflowx.core import get_metric
from optiflowx.optimizers import PSOOptimizer, GeneticOptimizer
from sklearn.metrics import mean_squared_error


def run_regression_sklearn_metric():
    X, y = make_regression(n_samples=200, n_features=8, noise=0.2, random_state=0)
    cfg = RandomForestConfig()
    wrapper = cfg.get_wrapper(task_type="regression")

    # Use the framework's normalized mse callable so ModelWrapper will
    # perform manual KFold evaluation rather than relying on sklearn's
    # scoring string (which expects 'neg_mean_squared_error').
    opt = PSOOptimizer(
        search_space=cfg.build_search_space(),
        metric="mse",
        custom_metric=get_metric("mse"),
        model_class=wrapper.model_class,
        X=X,
        y=y,
        n_particles=8,
    )
    best_params, best_score = opt.run(max_iters=3)
    print("regression + sklearn metric (mse) ->", best_score, best_params)


def run_regression_custom_metric():
    X, y = make_regression(n_samples=200, n_features=8, noise=0.2, random_state=1)
    cfg = RandomForestConfig()
    wrapper = cfg.get_wrapper(task_type="regression")

    def neg_rmse(y_true, y_pred):
        # avoid using sklearn squared kw to remain compatible across versions
        mse = mean_squared_error(y_true, y_pred)
        rmse = mse ** 0.5
        return -float(rmse)

    opt = GeneticOptimizer(
        search_space=cfg.build_search_space(),
        metric="mse",
        custom_metric=neg_rmse,
        model_class=wrapper.model_class,
        X=X,
        y=y,
        population=12,
    )
    best_params, best_score = opt.run(max_iters=3)
    print("regression + custom metric (neg_rmse) ->", best_score, best_params)


def run_custom_regression_sklearn_metric():
    X, y = make_regression(n_samples=150, n_features=6, noise=0.1, random_state=2)
    cfg = CustomModelConfig()
    # custom model wrapper may be used for regression too
    wrapper = cfg.get_wrapper()

    opt = PSOOptimizer(
        search_space=cfg.build_search_space(),
        metric="mse",
        custom_metric=get_metric("mse"),
        model_class=wrapper.model_class,
        X=X,
        y=y,
        n_particles=6,
    )
    best_params, best_score = opt.run(max_iters=3)
    print("custom regression model + sklearn metric ->", best_score, best_params)


def run_custom_regression_custom_metric():
    X, y = make_regression(n_samples=150, n_features=6, noise=0.1, random_state=3)
    cfg = CustomModelConfig()
    wrapper = cfg.get_wrapper()

    def neg_mae(y_true, y_pred):
        from sklearn.metrics import mean_absolute_error

        return -float(mean_absolute_error(y_true, y_pred))

    opt = GeneticOptimizer(
        search_space=cfg.build_search_space(),
        metric="mse",
        custom_metric=neg_mae,
        model_class=wrapper.model_class,
        X=X,
        y=y,
        population=8,
    )
    best_params, best_score = opt.run(max_iters=3)
    print("custom regression model + custom metric ->", best_score, best_params)


if __name__ == "__main__":
    print("== Regression examples ==")
    run_regression_sklearn_metric()
    run_regression_custom_metric()
    run_custom_regression_sklearn_metric()
    run_custom_regression_custom_metric()
