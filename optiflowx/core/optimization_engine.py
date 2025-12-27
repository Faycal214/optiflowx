import time
import logging
from optiflowx.models.registry import MODEL_REGISTRY
from .parallel_executor import ParallelExecutor
from optiflowx.optimizers import (
    GeneticOptimizer,
    PSOOptimizer,
    BayesianOptimizer,
    SimulatedAnnealingOptimizer,
    TPEOptimizer,
    RandomSearchOptimizer,
    GreyWolfOptimizer,
    AntColonyOptimizer,
)


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
        """Initialize an optimization engine for a given model and optimizer.

        Args:
            model_key (str): Key identifying the model in MODEL_REGISTRY.
            optimizer_key (str): Name of the optimizer (e.g., 'genetic', 'pso').
            dataset (tuple): Tuple of (X, y) for training and evaluation.
            metric (str or callable): Evaluation metric or custom metric function.
            strategy_params (dict, optional): Optimizer-specific parameters.
        """
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

        # Select optimizer
        # Pass metric and any custom_metric through to optimizer constructors
        opt_kwargs = dict(metric=self.metric, **self.strategy_params)
        if self.optimizer_key == "pso":
            self.optimizer = PSOOptimizer(self.search_space, **opt_kwargs)
        elif self.optimizer_key == "bayesian":
            self.optimizer = BayesianOptimizer(self.search_space, **opt_kwargs)
        elif self.optimizer_key == "genetic":
            self.optimizer = GeneticOptimizer(self.search_space, **opt_kwargs)
        elif self.optimizer_key == "simulated_annealing":
            self.optimizer = SimulatedAnnealingOptimizer(self.search_space, **opt_kwargs)
        elif self.optimizer_key == "tpe":
            self.optimizer = TPEOptimizer(self.search_space, **opt_kwargs)
        elif self.optimizer_key == "random_search":
            self.optimizer = RandomSearchOptimizer(self.search_space, **opt_kwargs)
        elif self.optimizer_key == "grey_wolf":
            self.optimizer = GreyWolfOptimizer(self.search_space, **opt_kwargs)
        elif self.optimizer_key == "ant_colony":
            self.optimizer = AntColonyOptimizer(self.search_space, **opt_kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_key}")

        # Ensure wrapper and optimizer know about any custom metric set later
        self.wrapper.custom_metric = None
        self.wrapper.task_type = getattr(self.wrapper, "task_type", "classification")

    def set_custom_metric(self, metric_fn):
        """Set a custom evaluation metric function.

        Args:
            metric_fn (callable): Function(y_true, y_pred) -> float.
        """
        self.custom_metric_fn = metric_fn
        # propagate to wrapper and optimizer if present
        try:
            self.wrapper.custom_metric = metric_fn
        except Exception:
            pass
        try:
            self.optimizer.custom_metric = metric_fn
        except Exception:
            pass

    def _evaluate_metric(self, y_true, y_pred):
        """Evaluate model predictions using the selected metric.

        Supports standard metrics (accuracy, f1, precision, recall, etc.)
        and user-supplied callable metrics.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels or probabilities.

        Returns:
            float: Computed metric score.
        """
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
                else:
                    return roc_auc_score(y_true, y_pred)
            elif m == "log_loss":
                from sklearn.metrics import log_loss

                return log_loss(y_true, y_pred)
            else:
                raise ValueError(
                    f"Unsupported metric: {self.metric}. Supported: accuracy, f1, precision, recall, roc_auc, log_loss or a custom callable."
                )
        except Exception:
            logging.exception("[Engine] Metric evaluation error")
            return float("-inf")

    def run(self, max_iters=10):
        """Execute the optimization process.

        Runs multiple iterations of candidate generation, evaluation,
        metric computation, and selection of the best configuration.

        Args:
            max_iters (int): Number of optimization iterations.

        Returns:
            tuple: (final_model, best_params, best_score)
        """
        X, y = self.dataset
        best = None
        total_start = time.time()

        logging.info("[INFO] Starting optimization for model: %s", self.model_key)
        logging.info(
            "[Engine] Running %s optimization for %s (metric=%s)",
            self.optimizer_key.upper(),
            self.model_key,
            self.metric,
        )
        metric = (
            self.custom_metric_fn if callable(self.custom_metric_fn) else self.metric
        )
        if callable(metric):
            logging.info("[Engine] Using custom metric function.")
        else:
            logging.info("[Engine] Using built-in metric: %s", str(metric).lower())

        for it in range(max_iters):
            iter_start = time.time()
            candidates = self.optimizer.suggest(None)
            # pass custom_metric and task_type to the executor so wrapper can use it
            results = self.executor.evaluate(
                candidates,
                self.wrapper,
                X,
                y,
                scoring=self.metric,
                custom_metric=self.custom_metric_fn,
                task_type=getattr(self.wrapper, "task_type", "classification"),
            )
            self.optimizer.update(results)

            for c in results:
                # Prefer the score produced by the executor/wrapper. The ParallelExecutor
                # (and ModelWrapper.train_and_score) already handles custom metrics and
                # returns negated error metrics for regression so the optimizer maximizes.
                if getattr(c, "score", None) is not None and c.score != float("-inf"):
                    # keep executor-provided score
                    pass
                else:
                    # Fallback: attempt to evaluate here using the engine's metric
                    try:
                        preds = c.model.predict(X)
                        score = self._evaluate_metric(y, preds)
                        c.score = score
                    except Exception:
                        logging.exception("[WARN] Evaluation failed for candidate")
                        c.score = float("-inf")
                if best is None or c.score > best.score:
                    best = c

            iter_time = time.time() - iter_start
            best_score = best.score if best is not None else float("-inf")
            logging.info("[Engine] Iter %d/%d | Best=%.4f | Time=%.2fs", it+1, max_iters, best_score, iter_time)

        total_time = time.time() - total_start
        logging.info("[Engine] Optimization finished in %.2fs", total_time)
        if best is not None:
            logging.info("[DONE] %s best_score=%.4f", self.model_key, best.score)
            final_params = best.params
            final_model = self.wrapper.model_class(**final_params)
            final_model.fit(X, y)
            return final_model, final_params, best.score
        else:
            logging.info("[DONE] %s best_score=None", self.model_key)
            return None, None, None
