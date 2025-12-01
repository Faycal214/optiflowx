from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import random
import logging
from optiflowx.core.base_optimizer import BaseOptimizer, Candidate

class GreyWolfOptimizer(BaseOptimizer):
    """
    Grey Wolf Optimizer (GWO) for hyperparameter optimization.

    This optimizer mimics the leadership hierarchy and hunting mechanism of grey wolves in nature.
    Supports mixed search spaces (continuous, discrete, categorical).

    Args:
        search_space: SearchSpace instance describing hyperparameters.
        metric: Evaluation metric (callable).
        model_class: ML model class (e.g., RandomForestClassifier).
        X: Training features.
        y: Training targets.
        population_size: Number of wolves in the population.
        max_iters: Maximum number of optimization iterations.
        random_state: Random seed for reproducibility.

    Returns:
        best_params (dict): Best hyperparameters found.
        best_score (float): Best evaluation score achieved.

    Example:
        >>> optimizer = GreyWolfOptimizer(search_space, metric, model_class, X, y)
        >>> best_params, best_score = optimizer.run()
    
    References:
        - Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey Wolf Optimizer. Advances in Engineering Software, 69, 46-61.
    """
    def __init__(
        self,
        search_space,
        metric: Callable,
        model_class: Any,
        X,
        y,
        population_size: int = 10,
        max_iters: int = 20,
        random_state: Optional[int] = None,
        custom_metric: Optional[Callable] = None,
        task_type: str = "classification",
    ):
        super().__init__(search_space, population_size)
        self.metric = metric
        self.model_class = model_class
        self.X = X
        self.y = y
        self.max_iters = max_iters
        self.random_state = random_state or np.random.randint(0, 10000)
        self.rng = np.random.RandomState(self.random_state)
        self.custom_metric = custom_metric
        self.task_type = task_type
        self.wrapper = None

    def _decode(self, vec: np.ndarray) -> Dict[str, Any]:
        params = {}
        for i, (name, info) in enumerate(self.search_space.parameters.items()):
            val = vec[i]
            if info["type"] == "categorical":
                idx = int(np.clip(round(val), 0, len(info["values"]) - 1))
                params[name] = info["values"][idx]
            elif info["type"] == "discrete":
                low, high = info["values"]
                params[name] = int(np.clip(round(val), low, high))
            else:
                low, high = info["values"]
                params[name] = float(np.clip(val, low, high))
        return params

    def _encode(self, param_dict: Dict[str, Any]) -> np.ndarray:
        vec = []
        for name, info in self.search_space.parameters.items():
            if info["type"] == "categorical":
                vec.append(float(info["values"].index(param_dict[name])))
            else:
                vec.append(float(param_dict[name]))
        return np.array(vec)

    def initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            params = self.search_space.sample()
            position = self._encode(params)
            self.population.append({"position": position, "candidate": Candidate(params)})

    def evaluate(self):
        import traceback
        from optiflowx.core.metrics import get_metric

        from optiflowx.core.metrics import get_metric

        for wolf in self.population:
            params = self._decode(wolf["position"])
            try:
                if getattr(self, "wrapper", None) is not None:
                    score, model = self.wrapper.train_and_score(
                        params, self.X, self.y, scoring=self.metric, custom_metric=self.custom_metric, task_type=self.task_type
                    )
                    score = float(score)
                else:
                    metric_fn = get_metric(self.custom_metric or self.metric)
                    model = self.model_class(**params)
                    model.fit(self.X, self.y)
                    preds = model.predict(self.X)
                    score = float(metric_fn(self.y, preds))
            except Exception:
                logging.exception("[Engine] Exception during GreyWolf evaluation")
                score = float("-inf")
                model = None
            wolf["candidate"].score = score
            wolf["candidate"].model = model if score != float("-inf") else None

    def run(self, max_iters=10) -> Tuple[Dict[str, Any], float]:
        import time
        self.initialize_population()
        dim = len(self.search_space.parameters)
        a = 2.0
        start_time = time.time()
        self._no_improve_count = 0
        self.best_candidate = None
        for i in range(max_iters):
            self.evaluate()
            sorted_wolves = sorted(self.population, key=lambda w: w["candidate"].score, reverse=True)
            alpha = sorted_wolves[0]["position"]
            beta = sorted_wolves[1]["position"] if len(sorted_wolves) > 1 else alpha
            delta = sorted_wolves[2]["position"] if len(sorted_wolves) > 2 else alpha
            for wolf in self.population:
                X = wolf["position"]
                for d in range(dim):
                    r1, r2 = self.rng.rand(), self.rng.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha[d] - X[d])
                    X1 = alpha[d] - A1 * D_alpha

                    r1, r2 = self.rng.rand(), self.rng.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta[d] - X[d])
                    X2 = beta[d] - A2 * D_beta

                    r1, r2 = self.rng.rand(), self.rng.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta[d] - X[d])
                    X3 = delta[d] - A3 * D_delta

                    X[d] = (X1 + X2 + X3) / 3.0
                wolf["position"] = X
            a = 2.0 - i * (2.0 / max_iters)
            best = max(self.population, key=lambda w: w["candidate"].score)
            if self.best_candidate is None or best["candidate"].score > self.best_candidate.score:
                self.best_candidate = best["candidate"]
                self._no_improve_count = 0
            else:
                self._no_improve_count += 1
            logging.info("[Engine] Iter %d/%d | Best=%.4f | Time=%.2fs", i+1, max_iters, self.best_candidate.score, time.time()-start_time)
            if self._no_improve_count >= getattr(self, "stagnation_limit", 10):
                logging.info("[Engine] Stopping early due to stagnation.")
                break
        logging.info("[Engine] Optimization finished in %.2fs", time.time()-start_time)
        return self.best_candidate.params, self.best_candidate.score
