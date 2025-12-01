# Getting Started

## Installation
Create a virtual environment and install:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install optiflowx
```

Or install directly from GitHub (latest version):

```bash
pip install -e git+https://github.com/<your-username>/optiflowx.git#egg=optiflowx
```

## Example: Quick Start

```python
from sklearn.datasets import make_classification
from optiflowx.optimizers.genetic import GeneticOptimizer
from optiflowx.models.configs.random_forest_config import RandomForestConfig

X, y = make_classification(n_samples=200, n_features=10, n_informative=8, random_state=42)

cfg = RandomForestConfig()
search_space = cfg.build_search_space()
model_class = cfg.get_wrapper().model_class

optimizer = GeneticOptimizer(
    search_space=search_space,
    metric="accuracy",
    model_class=model_class,
    X=X,
    y=y,
    population=10,
    mutation_prob=0.2,
)

best_params, best_score = optimizer.run(max_iters=5)
print("Best score:", best_score)
print("Best params:", best_params)
```

Quick test runs and CI: set environment variables to speed up examples during testing:

```bash
export EXAMPLES_FAST_MODE=1
export EXAMPLES_MAX_ITERS=1
```

This will make example scripts run a minimal fast pass (useful for CI).
