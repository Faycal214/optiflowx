from skopt import Optimizer as SkOptimizer
import warnings
from skopt.space import Real, Integer, Categorical
from optiflowx.core import get_metric
import logging


class BayesianOptimizer:
    """Bayesian optimization algorithm using scikit-optimize (skopt).

    This optimizer adaptively explores a search space to maximize an objective
    function (model performance metric). It builds a surrogate model to balance
    exploration of new configurations and exploitation of known good ones.

    Attributes:
        search_space: SearchSpace instance defining parameter ranges.
        metric (str or callable): Scoring metric or callable(y_true, y_pred).
        model_class (class): Model class to instantiate during evaluation.
        X (array-like): Training features.
        y (array-like): Training labels.
        n_initial_points (int): Number of random evaluations before using the surrogate.
        random_state (int): Random seed for reproducibility.
        stagnation_limit (int): Max iterations without improvement before early stopping.
        optimizer (skopt.Optimizer): Underlying Bayesian optimizer.
        trials (list): History of (x, y) tuples of sampled points and scores.
        results (list): List of evaluated Candidate instances.
        iteration (int): Current optimization iteration.
        best_candidate (Candidate): Best-performing configuration so far.
    """

    class Candidate:
        """Container for parameter sets evaluated during optimization."""

        def __init__(self, params):
            """Initialize a candidate configuration.

            Args:
                params (dict): Parameter dictionary for this candidate.
            """
            self.params = params
            self.score = None
            self.model = None

    def __init__(
        self,
        search_space,
        metric,
        model_class,
        X,
        y,
        n_initial_points=5,
        random_state=None,
        stagnation_limit=10,
        custom_metric=None,
        task_type: str = "classification",
    ):
        """Initialize the Bayesian optimizer.

        Args:
            search_space: SearchSpace defining model parameters to optimize.
            metric (str or callable): Metric name or callable(y_true, y_pred).
            model_class (class): Model class to optimize.
            X (array-like): Feature matrix.
            y (array-like): Target vector.
            n_initial_points (int, optional): Number of random initial evaluations. Defaults to 5.
            random_state (int, optional): Random seed. Defaults to 42.
            stagnation_limit (int, optional): Max no-improvement iterations before stopping. Defaults to 10.
        """
        self.search_space = search_space
        self.metric = metric
        self.custom_metric = custom_metric
        self.task_type = task_type
        self.model_class = model_class
        self.wrapper = None
        self.X = X
        self.y = y
        self.n_initial_points = n_initial_points
        self.random_state = random_state or 42
        self.sk_space = self._to_skopt_space()
        self.optimizer = SkOptimizer(
            dimensions=self.sk_space,
            random_state=self.random_state,
            n_initial_points=self.n_initial_points,
        )
        # Suppress skopt warning about duplicate evaluations (it injects random points).
        # This message is noisy in test runs and safe to ignore; it's an internal
        # skopt behavior when suggested points collide with previous ones.
        warnings.filterwarnings(
            "ignore",
            message=r"The objective has been evaluated at point .* before, using random point .*",
            category=UserWarning,
        )
        self.trials = []
        self.results = []
        self.iteration = 0
        self.best_candidate = None
        self.stagnation_limit = stagnation_limit
        self._no_improve_count = 0

    def initialize_population(self):
        """Reset optimizer state before a new run."""
        self.trials = []
        self.results = []
        self.iteration = 0
        self.best_candidate = None
        self._no_improve_count = 0

    def generate_candidates(self):
        """Generate candidate parameter configurations for evaluation.

        Returns:
            list of Candidate: New candidate configurations.
        """
        if len(self.trials) == 0:
            samples = [self.search_space.sample() for _ in range(self.n_initial_points)]
            return [self.Candidate(s) for s in samples]
        suggestions = self.optimizer.ask(n_points=1)
        param_names = list(self.search_space.parameters.keys())
        candidates = []
        for s in suggestions:
            cand_params = dict(zip(param_names, s))
            candidates.append(self.Candidate(cand_params))
        return candidates

    def evaluate_candidates(self, candidates):
        """Train and evaluate all generated candidates.

        Args:
            candidates (list of Candidate): Parameter configurations to evaluate.
        """
        # Prefer wrapper-based evaluation if a ModelWrapper was provided.
        for cand in candidates:
            try:
                if getattr(self, "wrapper", None) is not None:
                    score, model = self.wrapper.train_and_score(
                        cand.params, self.X, self.y, scoring=self.metric, custom_metric=self.custom_metric, task_type=self.task_type
                    )
                    cand.score = float(score)
                    cand.model = model
                else:
                    metric_fn = get_metric(self.custom_metric or self.metric)
                    model = self.model_class(**cand.params)
                    model.fit(self.X, self.y)
                    preds = model.predict(self.X)
                    cand.score = float(metric_fn(self.y, preds))
                    cand.model = model
            except Exception:
                cand.score = float("-inf")
                cand.model = None

    def update_state(self, candidates):
        """Update internal optimizer state with evaluated candidates.

        Args:
            candidates (list of Candidate): Evaluated candidates to register.
        """
        for cand in candidates:
            x = [cand.params[k] for k in self.search_space.parameters.keys()]
            y = -cand.score  # skopt minimizes the objective

            # Enforce valid bounds and handle categorical values
            x_clean = []
            for name, info in self.search_space.parameters.items():
                val = cand.params[name]
                t = info["type"]
                v = info["values"]
                if t == "continuous":
                    low, high = v
                    val = max(min(val, high), low)
                elif t in ["categorical", "discrete"]:
                    if val is None:
                        val = "none" if "none" in v else v[0]
                    if val not in v:
                        val = v[0]
                x_clean.append(val)

            # Update best candidate and stagnation counter
            if self.best_candidate is None or cand.score > self.best_candidate.score:
                self.best_candidate = cand
                self._no_improve_count = 0
            else:
                self._no_improve_count += 1

            # Register trial in Bayesian optimizer
            self.trials.append((x_clean, y))
            self.optimizer.tell(x_clean, y)

        # Track results and best candidate
        self.results.extend(candidates)
        best = max(
            candidates, key=lambda c: c.score if c.score is not None else float("-inf")
        )
        if self.best_candidate is None or best.score > self.best_candidate.score:
            self.best_candidate = best

    def run(self, max_iters=10):
        """Execute the Bayesian optimization loop.

        Args:
            max_iters (int, optional): Maximum number of optimization iterations. Defaults to 10.

        Returns:
            tuple:
                - best_params (dict): Best-found hyperparameters.
                - best_score (float): Corresponding evaluation score.
        """
        import time

        self.initialize_population()
        start_time = time.time()
        for i in range(max_iters):
            candidates = self.generate_candidates()
            self.evaluate_candidates(candidates)
            self.update_state(candidates)
            logging.info("[Engine] Iter %d/%d | Best=%.4f | Time=%.2fs", i+1, max_iters, self.best_candidate.score, time.time()-start_time)
            self.iteration += 1
            if self._no_improve_count >= self.stagnation_limit:
                logging.info("[Engine] Stopping early due to stagnation.")
                break

        logging.info("[Engine] Optimization finished in %.2fs", time.time()-start_time)
        if self.best_candidate is not None:
            return self.best_candidate.params, self.best_candidate.score
        return None, None

    def _to_skopt_space(self):
        """Convert SearchSpace definition into skopt-compatible space.

        Returns:
            list: List of skopt.space dimensions (Real, Integer, or Categorical).

        Raises:
            ValueError: If a parameter type is unsupported.
        """
        out = []
        for name, info in self.search_space.parameters.items():
            t = info["type"]
            v = info["values"]
            log = info.get("log", False)
            if t == "continuous":
                low, high = v
                if log and (low <= 0 or high <= 0):
                    log = False
                out.append(
                    Real(
                        low, high, prior="log-uniform" if log else "uniform", name=name
                    )
                )
            elif t == "int":
                low, high = v
                out.append(Integer(low, high, name=name))
            elif t in ["discrete", "categorical"]:
                out.append(Categorical(v, name=name))
            else:
                raise ValueError(f"Unknown parameter type {t} for {name}")
        return out
