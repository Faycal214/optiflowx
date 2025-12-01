import optuna
from optiflowx.core.metrics import get_metric
import logging


class TPEOptimizer:
    """Bayesian optimization engine using Optuna's TPE sampler.

    This class manages hyperparameter search by sampling candidates from
    a search space, training models, and updating the study with observed
    scores to guide future trials.

    Attributes:
        search_space: SearchSpace object defining parameter domains.
        metric: Callable metric function or string key for evaluation.
        model_class: Model class to instantiate and train.
        X: Training features.
        y: Training labels.
        population_size: Number of trials per iteration.
        study: optuna.study.Study instance.
        trials: List of tuples (params, score).
        iteration: Current optimization iteration.
        best_candidate: Best Candidate found so far.
        stagnation_limit: Max iterations without improvement before stopping.
        _no_improve_count: Counter for stagnation tracking.
    """

    class Candidate:
        """Container for model configuration and results.

        Attributes:
            params: Dict of hyperparameter values.
            score: Evaluation score for the candidate.
            model: Trained model instance.
            trial: Associated Optuna trial object.
        """

        def __init__(self, params):
            """Initialize a candidate with parameters.

            Args:
                params: Dict of model hyperparameters.
            """
            self.params = params
            self.score = None
            self.model = None
            self.trial = None

    def __init__(
        self,
        search_space,
        metric,
        model_class,
        X,
        y,
        population_size=10,
        stagnation_limit=10,
        custom_metric=None,
        task_type: str = "classification",
    ):
        """Initialize the TPE optimizer.

        Args:
            search_space: SearchSpace instance defining parameters.
            metric: Callable or string metric to optimize.
            model_class: Model class with fit and predict methods.
            X: Training features.
            y: Training labels.
            population_size: Number of candidates per generation.
            stagnation_limit: Number of iterations with no improvement allowed.
        """
        self.search_space = search_space
        self.metric = metric
        self.custom_metric = custom_metric
        self.task_type = task_type
        self.model_class = model_class
        self.wrapper = None
        self.X = X
        self.y = y
        self.population_size = population_size
        self.study = optuna.create_study(direction="maximize")
        self.trials = []
        self.iteration = 0
        self.best_candidate = None
        self.stagnation_limit = stagnation_limit
        self._no_improve_count = 0

    def initialize_population(self):
        """Reset internal state before running optimization."""
        self.trials = []
        self.iteration = 0
        self.best_candidate = None
        self._no_improve_count = 0

    def _define_search_space(self, trial):
        """Define parameters to be suggested by Optuna for one trial.

        Args:
            trial: optuna.trial.Trial object used for suggesting values.

        Returns:
            Dict of sampled hyperparameter values.
        """
        params = {}
        for name, cfg in self.search_space.parameters.items():
            ptype = cfg["type"]
            values = cfg["values"]
            log = cfg.get("log", False)
            if ptype in ("int", "discrete"):
                if (
                    isinstance(values, (list, tuple))
                    and len(values) == 2
                    and all(isinstance(x, int) for x in values)
                ):
                    params[name] = trial.suggest_int(
                        name, values[0], values[1], log=log
                    )
                else:
                    params[name] = trial.suggest_categorical(name, list(values))
            elif ptype in ("float", "continuous"):
                params[name] = trial.suggest_float(name, values[0], values[1], log=log)
            elif ptype == "categorical":
                params[name] = trial.suggest_categorical(name, list(values))
            else:
                raise ValueError(
                    f"[TPE] Unsupported parameter type: {ptype} for {name}"
                )
        return params

    def generate_candidates(self):
        """Generate a batch of candidate configurations.

        Returns:
            List of Candidate instances with suggested parameters.
        """
        candidates = []
        for _ in range(self.population_size):
            trial = self.study.ask()
            params = self._define_search_space(trial)
            candidate = self.Candidate(params=params)
            candidate.trial = trial
            candidates.append(candidate)
        return candidates

    def evaluate_candidates(self, candidates):
        """Train and evaluate all given candidates.

        Args:
            candidates: List of Candidate objects to evaluate.
        """
        metric_fn = get_metric(self.custom_metric or self.metric)
        for cand in candidates:
            try:
                if getattr(self, "wrapper", None) is not None:
                    score, model = self.wrapper.train_and_score(
                        cand.params, self.X, self.y, scoring=self.metric, custom_metric=self.custom_metric, task_type=self.task_type
                    )
                    cand.score = float(score)
                    cand.model = model
                else:
                    model = self.model_class(**cand.params)
                    model.fit(self.X, self.y)
                    preds = model.predict(self.X)
                    cand.score = float(metric_fn(self.y, preds))
                    cand.model = model
            except Exception:
                cand.score = float("-inf")
                cand.model = None

    def update_state(self, candidates):
        """Update study and internal best state after evaluating candidates.

        Args:
            candidates: List of evaluated Candidate objects.
        """
        for candidate in candidates:
            score = candidate.score
            trial = getattr(candidate, "trial", None)
            if trial is not None:
                self.study.tell(trial, score)
            if self.best_candidate is None or score > self.best_candidate.score:
                self.best_candidate = candidate
                self._no_improve_count = 0
            else:
                self._no_improve_count += 1
            self.trials.append((candidate.params, score))
        best = max(
            candidates, key=lambda c: c.score if c.score is not None else float("-inf")
        )
        if self.best_candidate is None or best.score > self.best_candidate.score:
            self.best_candidate = best

    def run(self, max_iters=10):
        """Run the optimization process.

        Args:
            max_iters: Maximum number of optimization iterations.

        Returns:
            Tuple (best_params, best_score) from the best candidate.
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
