---
title: Bayesian Optimization
sidebar_label: Bayesian Optimization
---

# Bayesian Optimization

Overview

Bayesian Optimization (BO) is a sample-efficient global optimization strategy for expensive, black-box functions. It is commonly used for hyperparameter tuning when each evaluation (training + validation) is costly.

Quick facts

- Type: Model-based (sequential) optimization
- Typical use: expensive-to-evaluate objective with <~20 dimensions
- Complexity: O(n^3) for Gaussian process updates (for GP-based BO), scales with number of observations
- Guarantees: No strict global guarantee for practical implementations; strong empirical performance

Core ideas

- Maintain a probabilistic surrogate model (e.g., Gaussian Process) over the objective.
- Use an acquisition function (EI, UCB, PI, Thompson) to propose the next evaluation point.
- Update the surrogate with the new observation and repeat until budget exhausted.

Common parameters

- `surrogate`: GP, random forest, or other regression model
- `acquisition`: EI / UCB / PI / Thompson sampling
- `n_initial_points`: number of random initialization evaluations

Usage example (OptiFlowX)

```python
from optiflowx.optimizers import BayesianOptimizer
from optiflowx.models.configs import RandomForestConfig
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
cfg = RandomForestConfig()
opt = BayesianOptimizer(search_space=cfg.build_search_space(), metric='accuracy', model_class=cfg.get_wrapper().model_class, X=X, y=y)
best, score = opt.run(max_iters=30)
print('best score', score)
```

Further reading

For high-dimensional or noisy objectives consider surrogate alternatives (e.g., random forests) or batched BO variants for parallel evaluations.
