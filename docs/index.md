---
title: Welcome to OptiFlowX
sidebar_label: Welcome to OptiFlowX
---

# 🧠 Welcome to OptiFlowX

OptiFlowX is an open-source optimization framework for machine learning, operations research, and applied AI. It provides a consistent API to run, compare, and track optimization workflows — from hyperparameter tuning to algorithm benchmarking.

## What you will find in this documentation

- Quickstart and getting started guides
- Examples with sklearn and custom models
- API reference for core building blocks
- Design system and theming notes for documentation

The site supports a dark-first design. For users coming from other themes, note that code-block contrast and semantic colors are optimized for comfortable reading on dark backgrounds.

## Quick installation

Install the stable release from PyPI:

```bash
pip install optiflowx
```

Or install the latest development version from the repository:

```bash
git clone https://github.com/Faycal214/optiflowx.git
cd optiflowx
pip install -e .
```

## Minimal example (Random Forest + GA)

```python
from sklearn.datasets import make_classification
from optiflowx.models.configs.random_forest_config import RandomForestConfig
from optiflowx.optimizers.genetic import GeneticOptimizer

X, y = make_classification(n_samples=200, n_features=12, random_state=0)
cfg = RandomForestConfig()
wrapper = cfg.get_wrapper(task_type="classification")

opt = GeneticOptimizer(
    search_space=cfg.build_search_space(),
    metric="accuracy",
    model_class=wrapper.model_class,
    X=X, y=y,
    population=10,
)

best_params, best_score = opt.run(max_iters=5)
print("Best score:", best_score)
print("Best params:", best_params)
```

## Goals & philosophy

OptiFlowX aims to be:

- Practical: run optimizers against real models and datasets
- Extensible: add new optimizers and model wrappers easily
- Reproducible: encourage deterministic experiments and CV-based evaluation

For detailed examples, API reference and contribution guidelines use the left-hand navigation.
