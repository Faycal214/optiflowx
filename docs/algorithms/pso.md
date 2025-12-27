---
title: Particle Swarm Optimization (PSO)
sidebar_label: PSO
---

# Particle Swarm Optimization (PSO)

Overview

Particle Swarm Optimization (PSO) is a population-based metaheuristic inspired by social behaviour (bird flocks, fish schools). It is well-suited for continuous, box-constrained optimization problems where derivatives are unavailable or expensive.

Quick facts

- Type: Stochastic population-based optimizer (swarm intelligence)
- Typical use: continuous hyperparameter tuning, low-to-moderate dimensional spaces
- Complexity: O(S * D * I) per evaluation round (S=swarm size, D=dimensions, I=iterations)
- Guarantees: No global optimum guarantee; empirically strong for many tasks

Core ideas

- Each particle keeps a position and a velocity in the search space.
- Velocities are updated using the particle's best-known position and the swarm's best-known position.
- The algorithm balances exploration (random components) and exploitation (attraction to known bests).

Common parameters

- `swarm_size` (S): number of particles
- `inertia` (w): controls momentum of particles
- `cognitive` (phi_p) and `social` (phi_g) coefficients: scale personal vs global attraction

Usage example (OptiFlowX)

```python
from optiflowx.optimizers import PSOOptimizer
from optiflowx.models.configs import RandomForestConfig
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
cfg = RandomForestConfig()
opt = PSOOptimizer(search_space=cfg.build_search_space(), metric='accuracy', model_class=cfg.get_wrapper().model_class, X=X, y=y, swarm_size=24)
best, score = opt.run(max_iters=20)
print('best score', score)
```

Further reading

See the original works by Kennedy & Eberhart and later surveys for deeper theoretical background. Practical PSO variants include topology changes (ring/local), inertia scheduling and constriction factors.
