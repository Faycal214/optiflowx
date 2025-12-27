---
title: Tree-structured Parzen Estimator (TPE)
sidebar_label: TPE
---

# Tree-structured Parzen Estimator (TPE)

Overview

TPE is a sequential model-based optimization algorithm that models P(x|y) using kernel density estimates and is a practical alternative to Gaussian Process-based BO, especially for mixed and conditional search spaces.

Quick facts

- Type: Sequential model-based optimization (SMBO)
- Typical use: mixed discrete/continuous spaces and conditional (tree-structured) search spaces
- Complexity: Depends on KDE fit cost; scales with number of observations and sampled candidates

Core ideas

- Partition observed trials into "good" and "bad" using a performance quantile.
- Fit density estimates `l(x)` (good) and `g(x)` (bad) and propose candidates maximizing `l(x)/g(x)`.

Common parameters

- `gamma`: quantile threshold separating good/bad trials
- `n_startup_trials`: initial random trials
- `n_candidates`: number of sampled candidates to evaluate the ratio

Usage example (OptiFlowX)

```python
from optiflowx.optimizers import TPEOptimizer
from optiflowx.models.configs import RandomForestConfig
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
cfg = RandomForestConfig()
opt = TPEOptimizer(search_space=cfg.build_search_space(), metric='accuracy', model_class=cfg.get_wrapper().model_class, X=X, y=y)
best, score = opt.run(max_iters=50)
print('best score', score)
```

Further reading

TPE is commonly used in practical hyperparameter frameworks (e.g., Hyperopt, Optuna) and handles conditional spaces gracefully.

sidebar_label: Tree-structured Parzen Estimator (TPE)
---

## Tree-structured Parzen Estimator (TPE)

TPE models P(x|y) using KDEs for good and bad trials and proposes candidates that maximize l(x)/g(x). It's effective for conditional and mixed search spaces.

### Components
- KDE-based densities l(x) and g(x)
- Threshold quantile γ separating good vs bad

### Pros / Cons
- Pros: handles conditional spaces, flexible
- Cons: KDE overhead on large histories
