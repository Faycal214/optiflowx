# API Reference

OptiFlowX provides a unified interface for running multiple optimization algorithms across different models and parameter spaces.
This page documents the main modules and their interactions.

---

## 🔧 Common Interface

Every optimizer in OptiFlowX follows a consistent interface for initialization and execution.

```python
from sklearn.tree import DecisionTreeClassifier
from optiflowx.optimizers.some_optimizer import SomeOptimizer
from optiflowx.models.configs.some_model_config import SomeModelConfig

# For Example we'll be using the decision tree model
model = DecisionTreeClassifier()

# Import the model's configurations for building the search space and get the model class
clf = SomeModelConfig()
search_space = clf.build_search_space()
model_class = clf.get_wrapper().model_class

# create the optimizer instance
optimizer = GeneticOptimizer(
    search_space=search_space,
    metric="accuracy",
    model_class=model_class,
    X=X_train,
    y=y_train,
    population=10,
    mutation_prob=0.3
)

# Train the optimizer and get the optimal solution
best_params, best_score = optimizer.run(max_iters=5)
print("OptiFlowX best:", best_score)

# Train the model with the optimal hyperparameters
model = model_class(**best_params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

## Arguments

| Argument | Description |
|-----------|--------------|
| **search_space** | Dictionary of hyperparameters and their ranges or choices. |
| **metric** | Evaluation metric (e.g., `"accuracy"`, `"f1"`, `"mse"`). |
| **model_class** | Model configuration class imported `from optiflowx.models.configs`. |
| **X, y** | Training data (features and labels). |
| **params** | Optional additional arguments (optimizer-specific). |

## Core Modules

### `optiflowx.core.search_space`

Handles definition and sampling of parameter spaces.

**Key Features:**

* Supports discrete, continuous, and categorical parameters.

* Provides random or structured sampling methods.

* Allows user-defined boundaries and constraints.

**Example:**

```python
from optiflowx.core.search_space import SearchSpace

search_space = SearchSpace()

search_space.add("n_estimators", "discrete", [20, 100])
search_space.add("max_depth", "discrete", [2, 20])
search_space.add("min_samples_split", "discrete", [2, 10])
search_space.add("min_samples_leaf", "discrete", [1, 5])
search_space.add("bootstrap", "categorical", [True, False])
search_space.add("max_features", "categorical", ["sqrt", "log2", None])

print(search_space.parameters)

sample = search_space.sample()
```

### `optiflowx.models.configs`

Contains configuration wrappers for different machine learning models.
Each config standardizes model initialization, fitting, and scoring so all optimizers can use them interchangeably.

**Example:**

```python
from optiflowx.models.configs.random_forest_config import RandomForestConfig

clf = RandomForestConfig()
model_class = clf.get_wrapper().model_class

print(model_class)
```

**Available Configs (examples)**:

* `random_forest_config.py`

* `xgboost_config.py`

* `svm_config.py`

* `mlp_config.py`

### `optiflowx.optimizers`

Implements multiple optimization algorithms, each extending a unified base optimizer interface.

**Available Optimizers**:

* `grid_search.py`

* `random_search.py`

* `genetic_algorithm.py`

* `simulated_annealing.py`

* `pso.py`

* `bayesian_optimization.py`

**Example usage**:

```python
from optiflowx.optimizers.genetic import GeneticOptimizer

optimizer = GeneticOptimizer(search_space, metric="accuracy", model_class=model_class, X=X, y=y, **optimizer_params)
best_params, best_score = optimizer.run(max_iters=20)
```

## Architecture Overview

OptiFlowX’s modular architecture separates concerns:

```mathematica
┌───────────────────────────────┐
│        User / API Layer       │
│   (OptiFlowX main interface)  │
└──────────────┬────────────────┘
               │
┌──────────────▼───────────────┐
│        Optimizers Layer      │
│ (GA, PSO, SA, Grid, Random)  │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│         Model Configs        │
│ (RandomForest, XGBoost, etc) │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│        Core Components       │
│  (Search Space, Utils, etc.) │
└───────────────────────────────┘
```

## Notes

* Every optimizer follows the same `.run()` entry point for reproducibility.

* Model configs ensure compatibility between algorithms and estimators.

* You can extend OptiFlowX by adding a new optimizer under `optiflowx/optimizers/` following the same structure.
