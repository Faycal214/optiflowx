import numpy as np


class PSOOptimizer:
    """Particle Swarm Optimization (PSO) optimizer for hyperparameter tuning.

    PSO is a population-based metaheuristic inspired by social behavior of birds or fish.
    Each "particle" represents a possible model configuration (set of hyperparameters).
    Particles move through the search space by combining their own best experiences
    with those of the overall swarm, gradually converging toward optimal solutions.

    Attributes:
        search_space (SearchSpace): Object defining the parameter types and ranges.
        metric (callable or str): Evaluation function or scorer name from sklearn.
        model_class (type): Model class (e.g., `RandomForestClassifier`).
        X (array-like): Training feature matrix.
        y (array-like): Target vector.
        n_particles (int): Number of particles in the swarm.
        w (float): Inertia weight controlling exploration vs exploitation.
        c1 (float): Cognitive coefficient (influence of particle's personal best).
        c2 (float): Social coefficient (influence of global best).
        velocity_threshold (float): Minimum velocity magnitude to trigger stagnation.
        stagnation_limit (int): Number of iterations allowed without improvement before stopping.
        global_best_position (np.ndarray or None): Best position found by the swarm.
        global_best_score (float): Best score achieved by any particle.
        iteration (int): Current iteration count.
        best_params (dict or None): Best decoded hyperparameter set found.
    """

    class Particle:
        """Represents one individual (particle) in the PSO swarm.

        Each particle tracks:
          - its current position (parameter vector)
          - its velocity
          - its own best historical position and score.

        Attributes:
            position (np.ndarray): Current position in the search space.
            velocity (np.ndarray): Current movement vector.
            best_position (np.ndarray): Best position this particle has seen.
            best_score (float): Best score this particle has achieved.
        """

        def __init__(self, position, velocity):
            self.position = position
            self.velocity = velocity
            self.best_position = position.copy()
            self.best_score = float("-inf")

    def __init__(
        self,
        search_space,
        metric,
        model_class,
        X,
        y,
        n_particles=20,
        w=0.7,
        c1=1.4,
        c2=1.4,
        velocity_threshold=1e-3,
        stagnation_limit=10,
    ):
        """Initializes the PSO optimizer.

        Args:
            search_space (SearchSpace): Defines hyperparameter search bounds.
            metric (callable or str): Metric or scorer to evaluate model performance.
            model_class (type): Model class to be instantiated and trained.
            X (array-like): Training features.
            y (array-like): Training targets.
            n_particles (int, optional): Number of particles. Defaults to 20.
            w (float, optional): Inertia weight for velocity. Defaults to 0.7.
            c1 (float, optional): Cognitive coefficient. Defaults to 1.4.
            c2 (float, optional): Social coefficient. Defaults to 1.4.
            velocity_threshold (float, optional): Stopping threshold. Defaults to 1e-3.
            stagnation_limit (int, optional): Early stop limit. Defaults to 10.
        """
        self.search_space = search_space
        self.metric = metric
        self.model_class = model_class
        self.X = X
        self.y = y
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.velocity_threshold = velocity_threshold
        self.stagnation_limit = stagnation_limit
        self.param_info = self.search_space.parameters
        self.particles = []
        self.global_best_position = None
        self.global_best_score = float("-inf")
        self._no_improve_count = 0
        self.iteration = 0
        self.best_params = None

    def initialize_population(self):
        """Initializes particles with random positions and zero velocities."""
        self.particles = []
        for _ in range(self.n_particles):
            p = self.search_space.sample()
            position = self._encode_position(p)
            velocity = np.zeros_like(position)
            self.particles.append(self.Particle(position, velocity))
        self.global_best_position = None
        self.global_best_score = float("-inf")
        self._no_improve_count = 0
        self.iteration = 0
        self.best_params = None

    def _encode_position(self, param_dict):
        """Encodes a parameter dictionary into a numerical vector.

        Args:
            param_dict (dict): Dictionary of hyperparameters.

        Returns:
            np.ndarray: Encoded numeric vector for internal computation.
        """
        vec = []
        for name, info in self.param_info.items():
            if info["type"] == "categorical":
                vec.append(float(info["values"].index(param_dict[name])))
            else:
                vec.append(float(param_dict[name]))
        return np.array(vec)

    def _decode_position(self, vec):
        """Decodes a numerical vector into a valid parameter dictionary.

        Args:
            vec (np.ndarray): Encoded position vector.

        Returns:
            dict: Dictionary of decoded parameter values.
        """
        params = {}
        for i, (name, info) in enumerate(self.param_info.items()):
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

    def evaluate_particles(self):
        """Trains and evaluates models for all particles.

        Each particle’s model is trained using its decoded parameters.
        The fitness (score) is then computed using the provided metric.
        """
        for particle in self.particles:
            params = self._decode_position(particle.position)
            try:
                model = self.model_class(**params)
                model.fit(self.X, self.y)
                preds = model.predict(self.X)
                if callable(self.metric):
                    score = self.metric(self.y, preds)
                else:
                    from sklearn.metrics import get_scorer

                    score = get_scorer(self.metric)(model, self.X, self.y)
            except Exception:
                score = float("-inf")
            if score > particle.best_score:
                particle.best_score = score
                particle.best_position = particle.position.copy()
            if score > self.global_best_score:
                self.global_best_score = score
                self.global_best_position = particle.position.copy()
                self.best_params = params

    def update_particles(self):
        """Updates all particle velocities and positions using PSO equations.

        Uses inertia (w), cognitive (c1), and social (c2) terms to update movement.
        Returns the best parameters and score found in the iteration.
        """
        for particle in self.particles:
            r1 = np.random.rand(len(particle.position))
            r2 = np.random.rand(len(particle.position))
            cognitive = self.c1 * r1 * (particle.best_position - particle.position)
            social = self.c2 * r2 * (self.global_best_position - particle.position)
            particle.velocity = self.w * particle.velocity + cognitive + social
            particle.position += particle.velocity

        # Evaluate updated particles and return the best one
        best_params = None
        best_score = float("-inf")
        for particle in self.particles:
            params = self._decode_position(particle.position)
            try:
                model = self.model_class(**params)
                model.fit(self.X, self.y)
                preds = model.predict(self.X)
                if callable(self.metric):
                    score = self.metric(self.y, preds)
                else:
                    from sklearn.metrics import get_scorer

                    score = get_scorer(self.metric)(model, self.X, self.y)
            except Exception:
                score = float("-inf")
            if score > best_score:
                best_score = score
                best_params = params
        if best_params is not None:
            print(f"Result for PSO: score={best_score:.4f}, params={best_params}")
            return best_params, best_score
        else:
            print("Result for PSO: score=-inf, params={}")
            return {}, float("-inf")

    def get_best_params(self):
        """Returns the best decoded parameter configuration found so far.

        Returns:
            dict: Decoded best parameters.
        """
        return self._decode_position(self.global_best_position)

    def run(self, max_iters=10):
        """Executes the PSO optimization loop.

        The algorithm runs up to `max_iters` iterations or stops early
        when no improvement occurs for several iterations.

        Args:
            max_iters (int, optional): Maximum iterations. Defaults to 10.

        Returns:
            tuple[dict, float]: Best parameter dictionary and score.
        """
        import time

        self.initialize_population()
        start_time = time.time()
        self.best_candidate = None
        for i in range(max_iters):
            self.evaluate_particles()

            # Identify current best performer
            best_score = float("-inf")
            best_params = None
            for particle in self.particles:
                params = self._decode_position(particle.position)
                try:
                    model = self.model_class(**params)
                    model.fit(self.X, self.y)
                    preds = model.predict(self.X)
                    if callable(self.metric):
                        score = self.metric(self.y, preds)
                    else:
                        from sklearn.metrics import get_scorer

                        score = get_scorer(self.metric)(model, self.X, self.y)
                except Exception:
                    score = float("-inf")
                if score > best_score:
                    best_score = score
                    best_params = params

            # Track best candidate
            if self.best_candidate is None or best_score > self.best_candidate.score:

                class Candidate:
                    def __init__(self, params, score):
                        self.params = params
                        self.score = score

                self.best_candidate = Candidate(best_params, best_score)

            self.update_particles()
            print(
                f"[Engine] Iter {i+1}/{max_iters} | Best={self.best_candidate.score:.4f} | Time={time.time()-start_time:.2f}s\n"
            )
            if self._no_improve_count >= self.stagnation_limit:
                print("[Engine] Stopping early due to stagnation.")
                break

        print(f"[Engine] Optimization finished in {time.time()-start_time:.2f}s")
        if self.best_candidate is not None:
            return self.best_candidate.params, self.best_candidate.score
        else:
            return {}, float("-inf")
