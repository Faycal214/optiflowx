---
title: Ant Colony Optimization (ACO)
sidebar_label: Ant Colony Optimization
---

# Ant Colony Optimization (ACO)

Overview

Ant Colony Optimization is a population-based metaheuristic inspired by pheromone communication in ant colonies. It is widely used for combinatorial optimization on graphs (e.g., shortest path, routing) and can be adapted to continuous problems.

Quick facts

- Type: Population-based pheromone-metaheuristic
- Typical use: graph-based combinatorial problems, routing, scheduling
- Complexity: O(S * L * I) where S=ants, L=path length, I=iterations

Core ideas

- Construct solutions stochastically guided by pheromone trails and heuristic desirability.
- Update pheromones globally and/or locally to reinforce good solutions and allow evaporation to avoid premature convergence.

Usage example (OptiFlowX)

```python
from optiflowx.optimizers import AntColonyOptimizer
from optiflowx.models.configs import RandomForestConfig
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
cfg = RandomForestConfig()
opt = AntColonyOptimizer(search_space=cfg.build_search_space(), metric='accuracy', model_class=cfg.get_wrapper().model_class, X=X, y=y)
best, score = opt.run(max_iters=40)
print('best score', score)
```

Further reading

ACO has many extensions (ACS, MMAS, rank-based, parallel variants). See algorithm surveys for practical parameter choices.

0 otherwise }

### Common extensions

#### Ant system (AS)

The first ACO algorithm, corresponding to the one above.

#### Ant colony system (ACS)

Modifies AS in three aspects:

- Biased edge selection toward exploitation.
- Local pheromone updates during solution construction.
- Only the best ant updates trails globally.

#### Elitist ant system

Global best solution deposits pheromone each iteration.

#### Max-min ant system (MMAS)

Controls max/min pheromone per trail, restricts deposition to the best ant, reinitializes trails at stagnation.

#### Rank-based ant system (ASrank)

Solutions ranked by length; best few update pheromone with weighted contributions.

#### Parallel ant colony optimization (PACO)

Partitions ants into groups with communication strategies.

#### Continuous orthogonal ant colony (COAC)

Uses orthogonal design and adaptive radius for strong global search.

#### Recursive ant colony optimization

Divides domain recursively into subdomains, promoting best solutions.

### Convergence

For some versions of the algorithm, it is possible to prove that it is convergent (i.e., it is able to find the global optimum in finite time). The first evidence of convergence for an ant colony algorithm was made in 2000, the graph-based ant system algorithm, and later on for the ACS and MMAS algorithms. Like most metaheuristics, it is very difficult to estimate the theoretical speed of convergence. A performance analysis of a continuous ant colony algorithm with respect to its various parameters (edge selection strategy, distance measure metric, and pheromone evaporation rate) showed that its performance and rate of convergence are sensitive to the chosen parameter values, and especially to the value of the pheromone evaporation rate. In 2004, Zlochin and his colleagues showed that ACO-type algorithms are closely related to stochastic gradient descent, Cross-entropy method and estimation of distribution algorithm. They proposed an umbrella term "Model-based search" to describe this class of metaheuristics.

### Article

Article : [Ant Colony Optimization: Artificial Ants as a Computational Intelligence Technique PDF](https://web.archive.org/web/20120222061542/http://iridia.ulb.ac.be/IridiaTrSeries/IridiaTr2006-023r001.pdf)
