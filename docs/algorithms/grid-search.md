---
title: Grid Search
sidebar_label: Grid Search
---

# Grid Search

Overview

Grid search exhaustively evaluates a discrete grid of hyperparameter combinations. It is straightforward but suffers from combinatorial explosion as dimensionality increases.

Quick facts

- Type: Exhaustive search
- Typical use: low-dimensional hyperparameter sweeps where discrete candidate sets are reasonable
- Complexity: O(Π_k n_k) across parameters (product of candidate counts) — grows exponentially with number of parameters

Core ideas

- Define finite candidate sets for each hyperparameter.
- Evaluate all combinations using cross-validation or a hold-out metric.
- Select the configuration with the best validation performance.

Usage example (OptiFlowX)

```python
from optiflowx.optimizers import GridSearchOptimizer
from optiflowx.models.configs import SVCConfig
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
cfg = SVCConfig()
opt = GridSearchOptimizer(search_space=cfg.build_search_space(), metric='accuracy', model_class=cfg.get_wrapper().model_class, X=X, y=y)
best, score = opt.run()
print('best score', score)
```
