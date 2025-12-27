"""Run all optimizers for classification scenarios.

This file contains the classification examples previously bundled in
`run_all_optimizers.py`. It runs each optimizer on a small synthetic
classification dataset with both sklearn and custom metrics, and with
both the library RandomForest wrapper and the provided CustomModel.
"""
import os
import sys
import logging
import traceback

# Ensure project root is importable when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sklearn.datasets import make_classification
import os
from optiflowx.models.configs import RandomForestConfig, CustomModelConfig
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


def classification_scenarios():
    print("\n== Classification scenarios ==")
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


if __name__ == "__main__":
    classification_scenarios()
"""Classification example scenarios for OptiFlowX.

Contains four examples (in one file):
- classification + sklearn metric
- classification + custom metric
- custom classification model + sklearn metric
- custom classification model + custom metric

These examples use small datasets and short runs for quick local testing.
"""
import os
import sys

# Ensure project root is on sys.path when running examples directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sklearn.datasets import make_classification
from optiflowx.models.configs import RandomForestConfig, CustomModelConfig
from optiflowx.optimizers import PSOOptimizer, GeneticOptimizer


def run_classification_sklearn_metric():
    X, y = make_classification(n_samples=200, n_features=8, random_state=0)
    cfg = RandomForestConfig()
    wrapper = cfg.get_wrapper(task_type="classification")

    opt = PSOOptimizer(
        search_space=cfg.build_search_space(),
        metric="accuracy",
        model_class=wrapper.model_class,
        X=X,
        y=y,
        n_particles=8,
    )
    best_params, best_score = opt.run(max_iters=3)
    print("classification + sklearn metric ->", best_score, best_params)


def run_classification_custom_metric():
    X, y = make_classification(n_samples=200, n_features=8, random_state=1)
    cfg = RandomForestConfig()
    wrapper = cfg.get_wrapper(task_type="classification")

    def my_accuracy(y_true, y_pred):
        # simple accuracy callable (higher is better)
        return float((y_true == y_pred).mean())

    opt = GeneticOptimizer(
        search_space=cfg.build_search_space(),
        metric="accuracy",
        custom_metric=my_accuracy,
        model_class=wrapper.model_class,
        X=X,
        y=y,
        population=12,
    )
    best_params, best_score = opt.run(max_iters=3)
    print("classification + custom metric ->", best_score, best_params)


def run_custom_model_sklearn_metric():
    X, y = make_classification(n_samples=150, n_features=6, random_state=2)
    cfg = CustomModelConfig()
    wrapper = cfg.get_wrapper()

    opt = PSOOptimizer(
        search_space=cfg.build_search_space(),
        metric="accuracy",
        model_class=wrapper.model_class,
        X=X,
        y=y,
        n_particles=6,
    )
    best_params, best_score = opt.run(max_iters=3)
    print("custom model + sklearn metric ->", best_score, best_params)


def run_custom_model_custom_metric():
    X, y = make_classification(n_samples=150, n_features=6, random_state=3)
    cfg = CustomModelConfig()
    wrapper = cfg.get_wrapper()

    def simple_score(y_true, y_pred):
        return float((y_true == y_pred).mean())

    opt = GeneticOptimizer(
        search_space=cfg.build_search_space(),
        metric="accuracy",
        custom_metric=simple_score,
        model_class=wrapper.model_class,
        X=X,
        y=y,
        population=8,
    )
    best_params, best_score = opt.run(max_iters=3)
    print("custom model + custom metric ->", best_score, best_params)


if __name__ == "__main__":
    print("== Classification examples ==")
    run_classification_sklearn_metric()
    run_classification_custom_metric()
    run_custom_model_sklearn_metric()
    run_custom_model_custom_metric()
