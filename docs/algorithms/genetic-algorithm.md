---
title: Genetic Algorithm (GA)
sidebar_label: Genetic Algorithm
---

# Genetic Algorithm (GA)

Overview

Genetic Algorithms are population-based evolutionary optimizers inspired by natural selection. They use mutation, crossover and selection operators to evolve candidate solutions over generations.

Quick facts

- Type: Evolutionary population-based optimizer
- Typical use: discrete and continuous optimization, combinatorial problems, hyperparameter search where recombination is useful
- Complexity: O(P * E * C) per generation (P=population size, E=evaluation cost, C=crossover/mutation cost)

Core ideas

- Maintain a population of candidate solutions encoded as chromosomes.
- Apply selection to favor fitter individuals.
- Use crossover (recombination) and mutation to generate new offspring.
- Repeat selection and variation for multiple generations.

Common parameters

- `population_size`: number of candidates per generation
- `mutation_prob`: per-gene mutation probability
- `crossover_rate`: probability of recombination between parents
- `selection`: tournament, roulette-wheel, rank-based

Usage example (OptiFlowX)

```python
from optiflowx.optimizers import GeneticOptimizer
from optiflowx.models.configs import RandomForestConfig
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
cfg = RandomForestConfig()
opt = GeneticOptimizer(search_space=cfg.build_search_space(), metric='accuracy', model_class=cfg.get_wrapper().model_class, X=X, y=y, population=40, mutation_prob=0.1)
best, score = opt.run(max_iters=50)
print('best score', score)
```

Practical tips

- Use elitism to retain the best individuals across generations.
- Tune mutation rate according to problem encoding — higher for discrete encodings.
- Combine with local search (memetic approaches) for improved convergence.
