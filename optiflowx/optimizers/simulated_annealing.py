import random
import math


class SimulatedAnnealingOptimizer:
    """Simulated Annealing optimization algorithm for hyperparameter tuning.

    This optimizer explores the search space by probabilistically accepting
    worse solutions to escape local minima. It gradually decreases the
    acceptance probability via a cooling schedule until convergence.

    Attributes:
        search_space: Object defining the parameter distributions to sample from.
        metric: Callable or string name of a sklearn scoring metric.
        model_class: Model constructor (e.g., sklearn estimator class).
        X: Training features.
        y: Training targets.
        population_size (int): Number of solutions maintained in each iteration.
        initial_temp (float): Starting temperature for simulated annealing.
        cooling_rate (float): Multiplicative cooling factor applied per iteration.
        mutation_rate (float): Probability of perturbing each parameter.
        t_min (float): Minimum temperature threshold for stopping.
        temperature (float): Current temperature value.
        scores (dict): Cached evaluation scores by candidate ID.
        iteration (int): Current iteration number.
        population (list): Current set of candidate solutions.
    """

    class Candidate:
        """Represents a single parameter configuration in the population."""

        def __init__(self, params):
            """Initialize a candidate.

            Args:
                params (dict): Parameter configuration of the candidate.
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
        population_size=10,
        initial_temp=1.0,
        cooling_rate=0.9,
        mutation_rate=0.3,
        t_min=1e-3,
    ):
        """Initialize the Simulated Annealing optimizer.

        Args:
            search_space: Object defining the parameter search space.
            metric: Callable or sklearn metric name used for evaluation.
            model_class: Model class to be instantiated (e.g., sklearn estimator).
            X: Training features.
            y: Training targets.
            population_size (int, optional): Number of candidate solutions. Defaults to 10.
            initial_temp (float, optional): Initial temperature. Defaults to 1.0.
            cooling_rate (float, optional): Cooling factor per iteration. Defaults to 0.9.
            mutation_rate (float, optional): Probability of mutating a parameter. Defaults to 0.3.
            t_min (float, optional): Minimum temperature for termination. Defaults to 1e-3.
        """
        self.search_space = search_space
        self.metric = metric
        self.model_class = model_class
        self.X = X
        self.y = y
        self.population_size = population_size
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.mutation_rate = mutation_rate
        self.t_min = t_min
        self.temperature = initial_temp
        self.scores = {}
        self.iteration = 0
        self.population = []

    def initialize_population(self):
        """Initialize the population of candidates and reset internal state."""
        self.population = [
            self.Candidate(self.search_space.sample())
            for _ in range(self.population_size)
        ]
        self.scores = {}
        self.temperature = self.initial_temp
        self.iteration = 0

    def evaluate_population(self):
        """Evaluate the current population by fitting and scoring models."""
        for cand in self.population:
            try:
                model = self.model_class(**cand.params)
                model.fit(self.X, self.y)
                preds = model.predict(self.X)
                if callable(self.metric):
                    score = self.metric(self.y, preds)
                else:
                    from sklearn.metrics import get_scorer

                    score = get_scorer(self.metric)(model, self.X, self.y)
                cand.score = score
                cand.model = model
            except Exception:
                cand.score = float("-inf")
                cand.model = None

    def update_state(self):
        """Update the population using the simulated annealing acceptance rule.

        Each candidate is perturbed and compared against its previous state.
        Worse candidates may be accepted with a probability proportional
        to `exp(-delta / temperature)` to encourage exploration.
        """
        # Store current scores
        for c in self.population:
            self.scores[id(c)] = c.score

        # Generate new candidates via random perturbation
        new_population = []
        for c in self.population:
            new_params = self._perturb(c.params)
            new_candidate = self.Candidate(new_params)
            new_population.append(new_candidate)

        # Acceptance check based on simulated annealing rule
        for new_c, old_c in zip(new_population, self.population):
            old_score = self.scores.get(id(old_c), float("inf"))
            new_score = self.scores.get(id(new_c), old_score)
            delta = new_score - old_score
            if delta < 0 or math.exp(-delta / self.temperature) > random.random():
                old_c.params = new_c.params
                old_c.score = new_score

        # Cool down temperature
        self.temperature *= self.cooling_rate

    def run(self, max_iters=10):
        """Run the full simulated annealing optimization loop.

        Args:
            max_iters (int, optional): Maximum number of iterations. Defaults to 10.

        Returns:
            tuple: (best_params, best_score)
                best_params (dict): Parameters of the best model found.
                best_score (float): Corresponding metric score.
        """
        import time

        self.initialize_population()
        start_time = time.time()
        best_score = float("-inf")
        self._no_improve_count = 0

        for i in range(max_iters):
            self.evaluate_population()
            self.update_state()
            scores = [c.score for c in self.population]
            current_best = max(scores) if scores else float("-inf")
            print(
                f"[Engine] Iter {i+1}/{max_iters} | Best={current_best:.4f} | Time={time.time()-start_time:.2f}s"
            )

            self.iteration += 1
            if current_best > best_score:
                best_score = current_best
                self._no_improve_count = 0
            else:
                self._no_improve_count += 1

            if self._no_improve_count >= getattr(self, "stagnation_limit", 10):
                print("[Engine] Stopping early due to stagnation.")
                break
            if self.temperature < self.t_min:
                break

        print(f"[Engine] Optimization finished in {time.time()-start_time:.2f}s")
        best = max(
            self.population,
            key=lambda c: c.score if c.score is not None else float("-inf"),
        )
        return best.params, best.score

    def _perturb(self, params):
        """Perturb a candidate’s parameters to explore new configurations.

        Args:
            params (dict): Current parameter configuration.

        Returns:
            dict: New parameter configuration after perturbation.
        """
        new_params = params.copy()
        for name, info in self.search_space.parameters.items():
            if random.random() < self.mutation_rate:
                t = info["type"]
                if t == "categorical":
                    new_params[name] = random.choice(info["values"])
                elif t == "discrete":
                    vals = info["values"]
                    if isinstance(vals, (list, tuple)):
                        new_params[name] = random.choice(vals)
                    else:
                        low, high = vals
                        new_params[name] = random.randint(low, high)
                elif t == "continuous":
                    low, high = info["values"]
                    log = info.get("log", False)
                    if log and (low > 0 and high > 0):
                        log_low, log_high = math.log(low), math.log(high)
                        new_val = math.exp(random.uniform(log_low, log_high))
                    else:
                        new_val = random.uniform(low, high)
                    new_params[name] = new_val
        return new_params
