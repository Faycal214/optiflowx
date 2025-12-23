# Getting Started

This page shows how to install OptiFlowX and run a quick example locally. The project is designed to be lightweight and easy to extend.

## 1. Install

We recommend using a virtual environment.

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install optiflowx
```

To install the latest development version from the repository:

```bash
pip install -e git+https://github.com/Faycal214/optiflowx.git#egg=optiflowx
```

Optional extras:

```bash
pip install dill  # enables serializing complex custom metrics for multiprocessing
pip install xgboost  # if you plan to use XGBoost configs/wrappers
```

## 2. Quickstart example (random forest + GA)

This example shows the minimal end-to-end flow: build a search space, create an optimizer, and run it.

```python
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
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
    population=12,
)

best_params, best_score = opt.run(max_iters=5)
print("Best score:", best_score)
print("Best parameters:", best_params)

# Fit final model with best parameters
final_model = wrapper.model_class(**best_params)
final_model.fit(X, y)
```

## 3. Fast example mode (CI)

Set these environment variables in CI to speed up examples and tests:

```bash
export EXAMPLES_FAST_MODE=1
export EXAMPLES_MAX_ITERS=3
```

These variables are read by the example scripts to shorten runtime while preserving correctness.

## 4. Troubleshooting

- If `multiprocessing` fails for a custom metric, install `dill` to enable robust serialization.
- Use `pip install -r requirements-test.txt` to install test dependencies.

For more examples and deeper configuration options, see the Examples and API pages.
