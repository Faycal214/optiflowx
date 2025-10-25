import time
from optiflowx.models.registry import MODEL_REGISTRY
from optiflowx.core.parallel_executor import ParallelExecutor
from optiflowx.optimizers.genetic import GeneticOptimizer
from optiflowx.optimizers.pso import PSOOptimizer
from optiflowx.optimizers.bayesian import BayesianOptimizer
from optiflowx.optimizers.simulated_annealing import SimulatedAnnealingOptimizer
from optiflowx.optimizers.tpe import TPEOptimizer
from optiflowx.optimizers.random_search import RandomSearchOptimizer


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

        cfg = MODEL_REGISTRY[model_key]
        self.search_space = cfg.build_search_space()
        self.wrapper = cfg.get_wrapper()
        self.executor = ParallelExecutor()

        # Select optimizer
        if self.optimizer_key == "pso":
            self.optimizer = PSOOptimizer(self.search_space, **self.strategy_params)
        elif self.optimizer_key == "bayesian":
            self.optimizer = BayesianOptimizer(
                self.search_space, **self.strategy_params
            )
        elif self.optimizer_key == "genetic":
            self.optimizer = GeneticOptimizer(self.search_space, **self.strategy_params)
        elif self.optimizer_key == "simulated_annealing":
            self.optimizer = SimulatedAnnealingOptimizer(
                self.search_space, **self.strategy_params
            )
        elif self.optimizer_key == "tpe":
            self.optimizer = TPEOptimizer(self.search_space, **self.strategy_params)
        elif self.optimizer_key == "random_search":
            self.optimizer = RandomSearchOptimizer(
                self.search_space, **self.strategy_params
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_key}")

    def set_custom_metric(self, metric_fn):
        """Set a custom evaluation metric function.

        Args:
            metric_fn (callable): Function(y_true, y_pred) -> float.
        """
        self.custom_metric_fn = metric_fn

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
            except Exception as e:
                print(f"[Engine] Custom metric error: {e}")
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
        except Exception as e:
            print(f"[Engine] Metric evaluation error: {e}")
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

        print(f"[INFO] Starting optimization for model: {self.model_key}")
        print(
            f"[Engine] Running {self.optimizer_key.upper()} optimization for {self.model_key} (metric={self.metric})"
        )
        metric = (
            self.custom_metric_fn if callable(self.custom_metric_fn) else self.metric
        )
        if callable(metric):
            print("[Engine] Using custom metric function.")
        else:
            print(f"[Engine] Using built-in metric: {str(metric).lower()}")

        for it in range(max_iters):
            iter_start = time.time()
            candidates = self.optimizer.suggest(None)
            results = self.executor.evaluate(candidates, self.wrapper, X, y)
            self.optimizer.update(results)

            for c in results:
                try:
                    preds = c.model.predict(X)
                    score = self._evaluate_metric(y, preds)
                    c.score = score
                except Exception as e:
                    print(f"[WARN] Evaluation failed: {e}")
                    c.score = float("-inf")
                if best is None or c.score > best.score:
                    best = c

            iter_time = time.time() - iter_start
            best_score = best.score if best is not None else float("-inf")
            print(
                f"[Engine] Iter {it+1}/{max_iters} | Best={best_score:.4f} | Time={iter_time:.2f}s"
            )

        total_time = time.time() - total_start
        print(f"[Engine] Optimization finished in {total_time:.2f}s")
        if best is not None:
            print(f"[DONE] {self.model_key} best_score={best.score:.4f}")
            final_params = best.params
            final_model = self.wrapper.model_class(**final_params)
            final_model.fit(X, y)
            return final_model, final_params, best.score
        else:
            print(f"[DONE] {self.model_key} best_score=None")
            return None, None, None
