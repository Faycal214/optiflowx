# core/base_optimizer.py
from typing import List, Dict, Any


class Candidate:
    """Container for a single candidate solution in population-based optimizers.

    Attributes:
        params (Dict[str, Any]): Hyperparameter dictionary for this candidate.
        score (float | None): Evaluation score. Higher is better for classification.
        model: Optionally store the trained/fitted estimator instance.
    """

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.score = None
        self.model = None  # Optionally store trained model


class BaseOptimizer:
    """Base class and standardized interface for population-based optimizers.

    This class defines a minimal, consistent API that all concrete optimizers
    should follow. The framework expects optimizers to manage a population of
    `Candidate` objects and to support iterative suggestion and update steps.

    Typical subclass responsibilities:
      - override `update(results)` to implement evolution/selection logic.
      - optionally override `initialize_population()` to customize seeding.
      - call `suggest()` each iteration to obtain candidates for evaluation.

    Attributes:
        search_space: SearchSpace instance describing hyperparameters.
        population_size (int): Number of candidates to maintain.
        population (List[Candidate]): Current population.
        iteration (int): Current iteration index (starts at 0).
        history (List[List[Candidate]]): Snapshot of populations per iteration.
        best_candidate (Candidate | None): Best candidate found so far.
        params (dict): Optimizer-specific kwargs (kept for convenience).
    """

    def __init__(self, search_space, population_size: int = 10, **kwargs):
        """Initialize base optimizer state.

        Args:
            search_space: SearchSpace instance used for sampling.
            population_size: Desired population size (default 10).
            **kwargs: Optional optimizer-specific parameters stored in `params`.
        """
        self.search_space = search_space
        self.population_size = population_size
        self.population: List[Candidate] = []
        self.iteration = 0
        self.history: List[List[Candidate]] = []  # Store population per iteration
        self.best_candidate: Candidate = None
        self.params = kwargs  # Store optimizer-specific parameters

    def initialize_population(self):
        """Create an initial population of candidates by sampling the search space.

        This method resets iteration counters, history and best candidate.
        Subclasses may override to implement custom seeding strategies.
        """
        self.population = [
            Candidate(self.search_space.sample()) for _ in range(self.population_size)
        ]
        self.iteration = 0
        self.history = []
        self.best_candidate = None

    def suggest(self, n: int = None) -> List[Candidate]:
        """Return the current population or a prefix of it for evaluation.

        Args:
            n: If provided return at most `n` candidates from the population.

        Returns:
            List[Candidate]: Candidates to evaluate this iteration.
        """
        if not self.population:
            self.initialize_population()
        if n is not None:
            return self.population[:n]
        return list(self.population)

    def update(self, results: List[Candidate]):
        """Update internal population based on evaluated results.

        Subclasses must implement this method. `results` is a list of
        Candidate objects whose `.score` (and optionally `.model`) have been set
        by the evaluator.

        Args:
            results: Evaluated Candidate list.
        """
        raise NotImplementedError

    def get_best(self) -> Candidate:
        """Return the best candidate found so far.

        Returns:
            Candidate or None if population is empty.
        """
        if not self.population:
            return None
        return max(
            self.population,
            key=lambda c: c.score if c.score is not None else float("-inf"),
        )

    def reset(self):
        """Reset optimizer state to start a fresh optimization run."""
        self.initialize_population()
