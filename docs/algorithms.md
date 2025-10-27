# Algorithms in OptiFlowX

## Genetic Algorithm (GA)
Population-based search using selection, crossover, and mutation.
Good for mixed (discrete + categorical) spaces.

## Particle Swarm Optimization (PSO)
Particles move in the search space using inertia and social behavior.
Efficient for continuous parameters.

## Simulated Annealing (SA)
Explores new solutions probabilistically, allowing worse ones early to escape local minima.

## Bayesian Optimization
Models the objective function using a surrogate (Gaussian Process) and selects promising regions to explore.

## Tree-structured Parzen Estimator (TPE)
Uses probability models to guide sampling. Handles conditional parameters better than pure Bayesian.

## Random Search
Samples random parameter sets. Simple but surprisingly effective for baselines.
