---
title: Random Search
sidebar_label: Random Search
---

# Random Search

Overview

Random Search samples hyperparameter configurations uniformly (or from user-specified distributions). It is simple, parallelizable and often a strong baseline — particularly when only a few hyperparameters dominate performance.

Quick facts

- Type: Stochastic sampling baseline
- Typical use: baseline comparisons, high-dimensional continuous spaces where grid is infeasible
- Complexity: O(N) evaluations where N is the number of sampled configurations

Core ideas

- Sample configurations from distributions (uniform, log-uniform, categorical).
- Evaluate each configuration independently (embarrassingly parallel). 
- Keep the best configuration according to the chosen metric.

Usage example (OptiFlowX)

```python
from optiflowx.optimizers import RandomSearchOptimizer
from optiflowx.models.configs import RandomForestConfig
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
cfg = RandomForestConfig()
opt = RandomSearchOptimizer(search_space=cfg.build_search_space(), metric='accuracy', model_class=cfg.get_wrapper().model_class, X=X, y=y)
best, score = opt.run(max_iters=50)
print('best score', score)
```
