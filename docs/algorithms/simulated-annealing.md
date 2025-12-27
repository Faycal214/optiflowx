---
title: Simulated Annealing (SA)
sidebar_label: Simulated Annealing
---

# Simulated Annealing (SA)

Overview

Simulated Annealing is a single-solution stochastic metaheuristic inspired by the physical annealing process. It is particularly useful for combinatorial and discrete optimization problems, but also applies to continuous domains.

Quick facts

- Type: Single-solution stochastic optimizer
- Typical use: combinatorial optimization, non-convex problems with many local optima
- Complexity: O(I * N_neighbours) per run (I=iterations)
- Guarantees: Convergence to a global optimum only under very slow cooling schedules (theoretical)

Core ideas

- Start from an initial solution and propose local moves.
- Accept worse moves with a probability that decreases with a temperature parameter to escape local optima.
- Gradually reduce temperature according to an annealing schedule.

Common parameters

- `initial_temperature` (T0): starting temperature
- `schedule`: temperature reduction policy (linear, exponential)
- `neighbour_fn`: function that proposes a nearby solution

Usage example (OptiFlowX)

```python
from optiflowx.optimizers import SimulatedAnnealingOptimizer
from optiflowx.models.configs import KNNConfig
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
cfg = KNNConfig()
opt = SimulatedAnnealingOptimizer(search_space=cfg.build_search_space(), metric='accuracy', model_class=cfg.get_wrapper().model_class, X=X, y=y)
best, score = opt.run(max_iters=200)
print('best score', score)
```

Further reading

Simulated annealing variants and cooling schedule design are key practical concerns; see classic texts for theoretical guarantees and practical schedules.
