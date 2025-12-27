---
title: Grey Wolf Optimization (GWO)
sidebar_label: Grey Wolf Optimization
---

# Grey Wolf Optimization (GWO)

Overview

Grey Wolf Optimization is a swarm intelligence metaheuristic inspired by the social hierarchy and hunting behaviour of grey wolves. It is an alternative population-based optimizer for continuous domains.

Quick facts

- Type: Population-based swarm metaheuristic
- Typical use: continuous search spaces where simple parameterization and few control parameters are desirable
- Complexity: O(S * D * I) per iteration (similar to other swarm methods)

Core ideas

- Simulate leadership ranks (alpha, beta, delta, omega) to guide search and exploitation.
- Combine encircling, hunting and attacking behaviours to move candidate solutions towards promising regions.

Usage example (OptiFlowX)

```python
from optiflowx.optimizers import GreyWolfOptimizer
from optiflowx.models.configs import RandomForestConfig
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
cfg = RandomForestConfig()
opt = GreyWolfOptimizer(search_space=cfg.build_search_space(), metric='accuracy', model_class=cfg.get_wrapper().model_class, X=X, y=y)
best, score = opt.run(max_iters=40)
print('best score', score)
```

Further reading

See Mirjalili's original paper and later surveys for parameter choices and hybridizations.
