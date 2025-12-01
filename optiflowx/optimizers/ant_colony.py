from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import random
import logging
from optiflowx.core.base_optimizer import BaseOptimizer, Candidate

class AntColonyOptimizer(BaseOptimizer):
    """
    Ant Colony Optimizer (ACO) for hyperparameter optimization.

    This optimizer simulates the foraging behavior of ants using pheromone trails and probabilistic solution construction.
    Supports mixed search spaces (continuous, discrete, categorical).

    Args:
        search_space: SearchSpace instance describing hyperparameters.
        metric: Evaluation metric (callable).
        model_class: ML model class (e.g., RandomForestClassifier).
        X: Training features.
        y: Training targets.
        colony_size: Number of ants in the colony.
        max_iters: Maximum number of optimization iterations.
        evaporation_rate: Pheromone evaporation rate (float).
        random_state: Random seed for reproducibility.

    Returns:
        best_params (dict): Best hyperparameters found.
        best_score (float): Best evaluation score achieved.

    Example:
        >>> optimizer = AntColonyOptimizer(search_space, metric, model_class, X, y)
        >>> best_params, best_score = optimizer.run()
    
    References:
        - Dorigo, M., & Stützle, T. (2004). Ant Colony Optimization. MIT Press.
    """
    def __init__(
        self,
        search_space,
        metric: Callable,
        model_class: Any,
        X,
        y,
        colony_size: int = 10,
        max_iters: int = 20,
        evaporation_rate: float = 0.2,
        random_state: Optional[int] = None,
        custom_metric: Optional[Callable] = None,
        task_type: str = "classification",
    ):
        super().__init__(search_space, colony_size)
        self.metric = metric
        self.model_class = model_class
        self.X = X
        self.y = y
        self.max_iters = max_iters
        self.evaporation_rate = evaporation_rate
        self.random_state = random_state or np.random.randint(0, 10000)
        self.rng = np.random.RandomState(self.random_state)
        self.dim = len(self.search_space.parameters)
        self.pheromone = np.ones([self.dim, 100])  # 100 bins per param
        self.custom_metric = custom_metric
        self.task_type = task_type
        self.wrapper = None

    def _decode(self, indices: List[int]) -> Dict[str, Any]:
        params = {}
        for i, (name, info) in enumerate(self.search_space.parameters.items()):
            if info["type"] == "categorical":
                params[name] = info["values"][indices[i] % len(info["values"])]
            elif info["type"] == "discrete":
                low, high = info["values"]
                params[name] = int(np.clip(low + indices[i], low, high))
            else:
                low, high = info["values"]
                bins = 100
                val = low + (high - low) * (indices[i] / (bins - 1))
                params[name] = float(val)
        return params

    def _sample_indices(self) -> List[int]:
        indices = []
        for i, info in enumerate(self.search_space.parameters.values()):
            bins = 100
            pher = self.pheromone[i]
            probs = pher / pher.sum()
            idx = self.rng.choice(np.arange(bins), p=probs)
            indices.append(idx)
        return indices

    def initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            indices = self._sample_indices()
            params = self._decode(indices)
            self.population.append({"indices": indices, "candidate": Candidate(params)})

    def evaluate(self):
        import traceback
        from optiflowx.core.metrics import get_metric

        from optiflowx.core.metrics import get_metric

        for ant in self.population:
            params = self._decode(ant["indices"])
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
            except Exception as e:
                logging.exception("[Engine] Exception during AntColony evaluation")
                score = float("-inf")
                model = None
            ant["candidate"].score = score
            ant["candidate"].model = model if score != float("-inf") else None

    def update_pheromone(self):
        self.pheromone *= (1.0 - self.evaporation_rate)
        best = max(self.population, key=lambda a: a["candidate"].score)
        for i, idx in enumerate(best["indices"]):
            self.pheromone[i, idx] += 1.0

    def run(self, max_iters=10) -> Tuple[Dict[str, Any], float]:
        import time
        self.initialize_population()
        start_time = time.time()
        self._no_improve_count = 0
        self.best_candidate = None
        for i in range(max_iters):
            self.evaluate()
            self.update_pheromone()
            best = max(self.population, key=lambda a: a["candidate"].score)
            if self.best_candidate is None or best["candidate"].score > self.best_candidate.score:
                self.best_candidate = best["candidate"]
                self._no_improve_count = 0
            else:
                self._no_improve_count += 1
            logging.info("[Engine] Iter %d/%d | Best=%.4f | Time=%.2fs", i+1, max_iters, self.best_candidate.score, time.time()-start_time)
            if self._no_improve_count >= getattr(self, "stagnation_limit", 10):
                logging.info("[Engine] Stopping early due to stagnation.")
                break
            self.population = []
            for _ in range(self.population_size):
                indices = self._sample_indices()
                params = self._decode(indices)
                self.population.append({"indices": indices, "candidate": Candidate(params)})
        logging.info("[Engine] Optimization finished in %.2fs", time.time()-start_time)
        return self.best_candidate.params, self.best_candidate.score
