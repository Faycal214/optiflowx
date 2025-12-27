"""Run all optimizers across classification and regression example scenarios.

This script programmatically creates the datasets and model wrappers used in
the examples and runs each optimizer for a few iterations to validate
compatibility across the codebase.
"""
import os
import sys
import traceback
import logging

# Ensure project root is importable when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Configure logging for consistent example output
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s")

from sklearn.datasets import make_classification, make_regression
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


def run_optimizer(opt_name, opt_cls, opt_kwargs, search_space, wrapper, X, y, metric="accuracy", custom_metric=None, task_type="classification"):
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
        # Propagate the ModelWrapper so optimizers can use consistent evaluation
        try:
            optimizer.wrapper = wrapper
        except Exception:
            pass
        best_params, best_score = optimizer.run(max_iters=3)
        # Ensure returned params are native Python types for logging/serialization
        try:
            best_params = sanitize_params(best_params)
        except Exception:
            # Fallback: if sanitizer fails, ignore and print original
            pass
        print(f"Result {opt_name}: score={best_score}, params={best_params}")
    except Exception:
        print(f"ERROR running {opt_name}:")
        traceback.print_exc()


def classification_scenarios():
    print("\n== Classification scenarios ==")
    # datasets and wrappers
    X1, y1 = make_classification(n_samples=150, n_features=8, random_state=0)
    rf_cfg = RandomForestConfig()
    rf_wrapper = rf_cfg.get_wrapper(task_type="classification")
    custom_cfg = CustomModelConfig()
    custom_wrapper = custom_cfg.get_wrapper()

    # scenario 1: classification + sklearn metric
    for name, cls, kwargs in OPTIMIZERS:
        run_optimizer(name, cls, kwargs, rf_cfg.build_search_space(), rf_wrapper, X1, y1, metric="accuracy", custom_metric=None, task_type="classification")

    # scenario 2: classification + custom metric
    def my_acc(y_true, y_pred):
        return float((y_true == y_pred).mean())

    for name, cls, kwargs in OPTIMIZERS:
        run_optimizer(name, cls, kwargs, rf_cfg.build_search_space(), rf_wrapper, X1, y1, metric="accuracy", custom_metric=my_acc, task_type="classification")

    # scenario 3: custom classification model + sklearn metric
    for name, cls, kwargs in OPTIMIZERS:
        run_optimizer(name, cls, kwargs, custom_cfg.build_search_space(), custom_wrapper, X1, y1, metric="accuracy", custom_metric=None, task_type="classification")

    # scenario 4: custom classification model + custom metric
    for name, cls, kwargs in OPTIMIZERS:
        run_optimizer(name, cls, kwargs, custom_cfg.build_search_space(), custom_wrapper, X1, y1, metric="accuracy", custom_metric=my_acc, task_type="classification")


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
    # Backwards-compatible: run both classification and regression examples.
    from examples.classification_examples import classification_scenarios
    from examples.regression_examples import regression_scenarios

    classification_scenarios()
    regression_scenarios()
