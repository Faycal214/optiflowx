import random
from optiflowx.core.metrics import get_metric
import logging


class GeneticOptimizer:
    """Genetic algorithm optimizer for hyperparameter tuning.

    This optimizer evolves a population of model configurations using
    selection, crossover, and mutation to find high-performing parameter sets.
    It supports early stopping when performance stagnates.

    Attributes:
        search_space (SearchSpace): Defines the possible parameter ranges.
        metric (callable or str): Evaluation metric or scorer name.
        model_class (type): Model class to instantiate with parameters.
        X (array-like): Training features.
        y (array-like): Training targets.
        population_size (int): Number of candidates per generation.
        elite_frac (float): Fraction of top performers retained each generation.
        crossover_prob (float): Probability of performing crossover.
        mutation_prob (float): Probability of mutating a candidate.
        rng (random.Random): Random number generator.
        stagnation_limit (int): Stop after this many iterations without improvement.
        best_candidate (Candidate or None): Current best-performing individual.
        iteration (int): Current iteration count.
    """

    class Candidate:
        """Represents a single solution (set of parameters) in the population.

        Attributes:
            params (dict): Model hyperparameters.
            score (float or None): Evaluation score.
            model (object or None): Trained model instance.
        """

        def __init__(self, params):
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
        population=20,
        elite_frac=0.2,
        crossover_prob=0.8,
        mutation_prob=0.2,
        seed=None,
        stagnation_limit=10,
        custom_metric=None,
        task_type: str = "classification",
    ):
        """Initializes the genetic optimizer.

        Args:
            search_space (SearchSpace): Defines hyperparameter bounds/types.
            metric (callable or str): Metric function or sklearn scorer name.
            model_class (type): ML model class (e.g., RandomForestClassifier).
            X (array-like): Training feature matrix.
            y (array-like): Training target vector.
            population (int, optional): Population size. Defaults to 20.
            elite_frac (float, optional): Fraction of elites to keep. Defaults to 0.2.
            crossover_prob (float, optional): Probability of crossover. Defaults to 0.8.
            mutation_prob (float, optional): Probability of mutation. Defaults to 0.2.
            seed (int, optional): Random seed. Defaults to None.
            stagnation_limit (int, optional): Early stop threshold. Defaults to 10.
        """
        self.search_space = search_space
        self.metric = metric
        self.custom_metric = custom_metric
        self.task_type = task_type
        self.model_class = model_class
        self.X = X
        self.y = y
        self.population_size = population
        self.elite_frac = elite_frac
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.rng = random.Random(seed)
        self.stagnation_limit = stagnation_limit
        self._no_improve_count = 0
        self._best_score = None
        self.population = []
        self.best_candidate = None
        self.iteration = 0
        self.wrapper = None

    def initialize_population(self):
        """Initializes the population with random candidates."""
        self.population = [
            self.Candidate(self.search_space.sample())
            for _ in range(self.population_size)
        ]
        self._no_improve_count = 0
        self._best_score = None
        self.best_candidate = None
        self.iteration = 0

    def evaluate_population(self):
        """Trains and evaluates each candidate model in the population.

        Each candidate’s model is trained using its parameters, and its
        score is computed using the provided metric function or scorer.
        """
        for cand in self.population:
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
                    score = metric_fn(self.y, preds)
                    cand.score = float(score)
                    cand.model = model
            except Exception:
                cand.score = float("-inf")
                cand.model = None

    def select_elites(self):
        """Selects the top-performing individuals in the population.

        Returns:
            list[Candidate]: The top fraction of individuals sorted by score.
        """
        sorted_pop = sorted(
            self.population,
            key=lambda c: c.score if c.score is not None else float("-inf"),
            reverse=True,
        )
        elite_n = max(1, int(self.elite_frac * len(sorted_pop)))
        return sorted_pop[:elite_n]

    def crossover(self, parent1, parent2):
        """Performs crossover between two parent candidates.

        Args:
            parent1 (Candidate): First parent.
            parent2 (Candidate): Second parent.

        Returns:
            Candidate: New child candidate with mixed parameters.
        """
        child_params = {}
        for k in parent1.params:
            child_params[k] = (
                parent1.params[k] if self.rng.random() < 0.5 else parent2.params[k]
            )
        return self.Candidate(child_params)

    def mutate(self, candidate):
        """Applies mutation to a candidate by randomly altering parameters.

        Args:
            candidate (Candidate): The candidate to mutate.

        Returns:
            Candidate: A new mutated candidate.
        """
        params = candidate.params.copy()
        sample_vals = self.search_space.sample()
        for k, v in params.items():
            if self.rng.random() < self.mutation_prob:
                # Use sampled value if available, otherwise keep existing
                params[k] = sample_vals.get(k, v)
        return self.Candidate(params)

    def run(self, max_iters=10):
        """Executes the genetic optimization process.

        The optimizer evolves the population for up to `max_iters`
        generations or stops early if performance stagnates.

        Args:
            max_iters (int, optional): Maximum number of generations. Defaults to 10.

        Returns:
            tuple[dict, float]: Best parameters and corresponding score.
        """
        import time

        self.initialize_population()
        start_time = time.time()
        for i in range(max_iters):
            self.evaluate_population()
            elites = self.select_elites()
            new_population = elites.copy()
            while len(new_population) < self.population_size:
                if len(elites) < 2:
                    parents_pool = self.population
                else:
                    parents_pool = elites

                if self.rng.random() < self.crossover_prob:
                    p1, p2 = self.rng.sample(parents_pool, 2)
                    child = self.crossover(p1, p2)
                else:
                    child = self.rng.choice(parents_pool)

                child = self.mutate(child)
                new_population.append(child)

            self.population = new_population
            best = max(
                self.population,
                key=lambda c: c.score if c.score is not None else float("-inf"),
            )
            if self.best_candidate is None or best.score > self.best_candidate.score:
                self.best_candidate = best
                self._no_improve_count = 0
            else:
                self._no_improve_count += 1

            logging.info("[Engine] Iter %d/%d | Best=%.4f | Time=%.2fs", i+1, max_iters, self.best_candidate.score, time.time()-start_time)
            if self._no_improve_count >= self.stagnation_limit:
                logging.info("[Engine] Stopping early due to stagnation.")
                break

        logging.info("[Engine] Optimization finished in %.2fs", time.time()-start_time)
        return self.best_candidate.params, self.best_candidate.score
