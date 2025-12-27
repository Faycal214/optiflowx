import time
import logging
from .parallel_executor import ParallelExecutor


class OptimizationEngine:
    """Main engine that coordinates model optimization.

    Handles the interaction between models, optimizers, datasets, and metrics.
    Executes optimization iterations and tracks the best configuration found.
    """

    def __init__(
        self,
        model_key: str,
        optimizer_key: str = "genetic",
        dataset=None,
        metric="accuracy",
        strategy_params=None,
        task_type: str = "classification",
    ):
        """Initialize an optimization engine for a given model and optimizer."""
        # ---------------------------------------
        # LAZY IMPORT (break circular dependency)
        # ---------------------------------------
        from optiflowx.models.registry import MODEL_REGISTRY

        self.model_key = model_key
        self.optimizer_key = optimizer_key.lower()
        self.dataset = dataset
        self.metric = metric
        self.custom_metric_fn = None
        self.strategy_params = strategy_params or {}
        self.task_type = task_type

        cfg = MODEL_REGISTRY[model_key]
        self.search_space = cfg.build_search_space()
        self.wrapper = cfg.get_wrapper(task_type=self.task_type)
        self.executor = ParallelExecutor()

        # ---------------------------------------
        # Optimizer selection (LAZY IMPORTS)
        # ---------------------------------------
        opt_kwargs = dict(metric=self.metric, **self.strategy_params)

        if self.optimizer_key == "genetic":
            from optiflowx.optimizers.genetic import GeneticOptimizer
            self.optimizer = GeneticOptimizer(self.search_space, **opt_kwargs)

        elif self.optimizer_key == "pso":
            from optiflowx.optimizers.pso import PSOOptimizer
            self.optimizer = PSOOptimizer(self.search_space, **opt_kwargs)

        elif self.optimizer_key == "bayesian":
            from optiflowx.optimizers.bayesian import BayesianOptimizer
            self.optimizer = BayesianOptimizer(self.search_space, **opt_kwargs)

        elif self.optimizer_key == "simulated_annealing":
            from optiflowx.optimizers.simulated_annealing import (
                SimulatedAnnealingOptimizer,
            )
            self.optimizer = SimulatedAnnealingOptimizer(
                self.search_space, **opt_kwargs
            )

        elif self.optimizer_key == "tpe":
            from optiflowx.optimizers.tpe import TPEOptimizer
            self.optimizer = TPEOptimizer(self.search_space, **opt_kwargs)

        elif self.optimizer_key == "random_search":
            from optiflowx.optimizers.random_search import RandomSearchOptimizer
            self.optimizer = RandomSearchOptimizer(self.search_space, **opt_kwargs)

        elif self.optimizer_key == "grey_wolf":
            from optiflowx.optimizers.grey_wolf import GreyWolfOptimizer
            self.optimizer = GreyWolfOptimizer(self.search_space, **opt_kwargs)

        elif self.optimizer_key == "ant_colony":
            from optiflowx.optimizers.ant_colony import AntColonyOptimizer
            self.optimizer = AntColonyOptimizer(self.search_space, **opt_kwargs)

        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_key}")

        self.wrapper.custom_metric = None
        self.wrapper.task_type = getattr(self.wrapper, "task_type", self.task_type)

    def set_custom_metric(self, metric_fn):
        """Set a custom evaluation metric function."""
        self.custom_metric_fn = metric_fn
        if hasattr(self.wrapper, "custom_metric"):
            self.wrapper.custom_metric = metric_fn
        if hasattr(self.optimizer, "custom_metric"):
            self.optimizer.custom_metric = metric_fn

    def _evaluate_metric(self, y_true, y_pred):
        """Evaluate model predictions using the selected metric."""
        metric = (
            self.custom_metric_fn if callable(self.custom_metric_fn) else self.metric
        )

        if callable(metric):
            try:
                return metric(y_true, y_pred)
            except Exception:
                logging.exception("[Engine] Custom metric error")
                return float("-inf")

        m = str(metric).lower()
        try:
            if m == "accuracy":
                from sklearn.metrics import accuracy_score
                return accuracy_score(y_true, y_pred)

            elif m == "f1":
                from sklearn.metrics import f1_score
                return f1_score(y_true, y_pred, average="weighted")

            elif m == "precision":
                from sklearn.metrics import precision_score
                return precision_score(y_true, y_pred, average="weighted")

            elif m == "recall":
                from sklearn.metrics import recall_score
                return recall_score(y_true, y_pred, average="weighted")

            elif m == "roc_auc":
                from sklearn.metrics import roc_auc_score
                if hasattr(y_pred, "shape") and len(y_pred.shape) > 1:
                    return roc_auc_score(y_true, y_pred, multi_class="ovr")
                return roc_auc_score(y_true, y_pred)

            elif m == "log_loss":
                from sklearn.metrics import log_loss
                return log_loss(y_true, y_pred)

            else:
                raise ValueError(f"Unsupported metric: {self.metric}")

        except Exception:
            logging.exception("[Engine] Metric evaluation error")
            return float("-inf")

    def run(self, max_iters=10):
        """Execute the optimization process."""
        X, y = self.dataset
        best = None
        start = time.time()

        for it in range(max_iters):
            candidates = self.optimizer.suggest(None)
            results = self.executor.evaluate(
                candidates,
                self.wrapper,
                X,
                y,
                scoring=self.metric,
                custom_metric=self.custom_metric_fn,
                task_type=getattr(self.wrapper, "task_type", self.task_type),
            )

            self.optimizer.update(results)

            for c in results:
                if getattr(c, "score", None) is None:
                    try:
                        preds = c.model.predict(X)
                        c.score = self._evaluate_metric(y, preds)
                    except Exception:
                        c.score = float("-inf")

                if best is None or c.score > best.score:
                    best = c

            logging.info(
                "[Engine] Iter %d/%d | Best=%.4f",
                it + 1,
                max_iters,
                best.score if best else float("-inf"),
            )

        logging.info("[Engine] Finished in %.2fs", time.time() - start)

        if best is None:
            return None, None, None

        final_model = self.wrapper.model_class(**best.params)
        final_model.fit(X, y)
        return final_model, best.params, best.score
