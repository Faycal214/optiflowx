import importlib
import inspect
import pkgutil
from typing import Dict, Any, Tuple
from optiflowx.models.registry import MODEL_REGISTRY
from .optimization_engine import OptimizationEngine
import logging


class ModelSearchManager:
    """Manages model hyperparameter search across registered model configurations.

    Automatically loads available models, initializes an optimization engine,
    and executes searches for one or multiple models using a chosen optimization strategy.
    """

    def __init__(
        self,
        models_package: str = "models",
        strategy: str = "genetic",
        n_samples: int = 50,
        scoring: str = "accuracy",
        cv: int = 3,
        n_jobs: int = -1,
        strategy_params: dict = None,
        custom_metric_fn=None,
    ):
        """Initialize a model search manager.

        Args:
            models_package (str): Name of the Python package containing model definitions.
            strategy (str): Optimization strategy key (e.g., 'genetic').
            n_samples (int): Number of samples or candidates to test per model.
            scoring (str): Evaluation metric name (e.g., 'accuracy', 'f1').
            cv (int): Number of cross-validation folds.
            n_jobs (int): Number of parallel jobs (-1 means all cores).
            strategy_params (dict, optional): Extra parameters for the optimizer.
            custom_metric_fn (Callable, optional): Custom scoring function (y_true, y_pred) -> float.
        """
        self.models_package = models_package
        self.strategy = strategy
        self.n_samples = n_samples
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = None if n_jobs == -1 else n_jobs
        self.strategy_params = strategy_params or {}
        self.custom_metric_fn = custom_metric_fn
        self.model_configs = self._load_all_configs()

    def _load_all_configs(self) -> Dict[str, Any]:
        """Load all model configurations from the specified models package.

        Returns:
            Dict[str, Any]: Mapping from model name to configuration class.
        """
        package = importlib.import_module(self.models_package)
        configs = {}
        for _, modname, _ in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"{self.models_package}.{modname}")
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if hasattr(obj, "name") and hasattr(obj, "build_search_space"):
                    configs[obj.name] = obj
        return configs

    def search_model(self, model_name: str, dataset: Tuple, max_iters: int = 10):
        """Run hyperparameter optimization for a single model.

        Args:
            model_name (str): Name of the model to optimize.
            dataset (Tuple): Tuple of (X, y) data.
            max_iters (int): Number of optimization iterations.

        Returns:
            dict: Contains best model instance, parameters, and score.
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")

        logging.info("\n[INFO] Starting optimization for model: %s", model_name)
        engine = OptimizationEngine(
            model_key=model_name,
            optimizer_key=self.strategy,
            dataset=dataset,
            metric=self.scoring,
            strategy_params=self.strategy_params,
        )

        if self.custom_metric_fn:
            engine.set_custom_metric(self.custom_metric_fn)

        model, params, score = engine.run(max_iters=max_iters)
        if score is not None:
            logging.info("[DONE] %s best_score=%.4f", model_name, score)
        else:
            logging.info("[DONE] %s best_score=None", model_name)
        return {"model": model, "params": params, "score": score}

    def search_all(self, dataset: Tuple, max_iters: int = 10):
        """Run optimization for all registered models.

        Args:
            dataset (Tuple): Tuple of (X, y) data.
            max_iters (int): Number of optimization iterations per model.

        Returns:
            list[dict]: Sorted list of model results with their scores.
        """
        results = []
        for model_name in MODEL_REGISTRY.keys():
            res = self.search_model(model_name, dataset, max_iters)
            res["model_name"] = model_name
            results.append(res)

        results = sorted(results, key=lambda r: r["score"], reverse=True)
        logging.info("\n[SUMMARY] Best models:")
        for rank, r in enumerate(results, 1):
            logging.info("%d. %s -> score=%.4f", rank, r["model_name"], r["score"])
        return results
