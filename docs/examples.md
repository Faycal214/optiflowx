# 🧪 Examples

This section demonstrates how to use **OptiFlowX** for hyperparameter optimization across different use cases.

We show :

1. **Automatic setup** — the library builds the search space and metric automatically.
2. **Custom setup** — the user defines their own search space and metric.
3. **Deep learning setup** — using PyTorch models inside OptiFlowX.

---

## Example 1 — Automatic Setup (Auto-Search Space + Metric)

OptiFlowX can automatically infer the parameter search space, model configuration, and default metric from the model type.

```python
from optiflowx.optimizers import BayesianOptimizer
from optiflowx.models.configs import RandomForestConfig
from sklearn.ensemble import RandomForestClassifier

# Load dataset
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)

# Load model configuration and search space
cfg = RandomForestConfig()
search_space = cfg.build_search_space()
model_class = cfg.get_wrapper().model_class

# AutoOptimizer chooses sensible defaults based on model_class
optimizer = opt_class(
        search_space=search_space,
        metric="accuracy",
        model_class=model_class, # model_class = RandomForestClassifier
        X=X,
        y=y,
        **opt_params,
    )

best_params, best_score = optimizer.run(max_iters=10)

print("Best Parameters:", best_params)
print("Best Score:", best_score)
```

### How it works

* `model_class` loads `RandomForestClassifier`.

* `metric` sets the default metric to accuracy.

* Search space is auto-generated from known model hyperparameters using `cfg.build_search_space()` from `RandomForestConfig()`.

---

## Example 2 — Custom Search Space and Metric

For full control, you can define your own search space and evaluation metric.
This allows adapting OptiFlowX to any scikit-learn compatible model.

```python
from optiflowx.optimizers import GeneticOptimizer
from optiflowx.core import SearchSpace
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.datasets import load_iris

# Load dataset
X, y = load_iris(return_X_y=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

# Import the model
model = RandomForestClassifier()

# Custom metric
def my_metric():
    return f1_score(y_true, y_pred)

# Costum Search space
search_space = SearchSpace()

# Complete the Search space with any hyperparameter
# Each one of the .add() method must contain :
# 1. hyperparameter name
# 2. type : ["discrete", "continuos", "categorical"]
# 3. range of values
search_space.add("n_estimators", "discrete", [20, 100])
search_space.add("max_depth", "discrete", [2, 20])
search_space.add("min_samples_split", "discrete", [2, 10])
search_space.add("min_samples_leaf", "discrete", [1, 5])
search_space.add("bootstrap", "categorical", [True, False])
search_space.add("max_features", "categorical", ["sqrt", "log2", None])

# Show the parameters
print(search_space.parameters)

# We can sample a solution condidate of hyperparameters
sample = serach_space.sample()
print(sample)

# Create the optimizer
optimizer = GeneticOptimizer(
    search_space=search_space,
    metric=my_metric(),
    model_class=RandomForestClassifier,
    X=X_train,
    y=y_train,
    population=10,
    mutation_prob=0.3
)

# Get the final results
best_params, best_score = optimizer.run(max_iters=5)
print("OptiFlowX best:", best_score)

# Display the optimal parameters
print(best_params)

# Train the model with the final results
model = RandomForestClassifier(**best_params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

### Key points

* User defines a custom metric (`f1_score` in this case).

* Works with any scikit-learn model, not only pre-defined configs.

* Same `.run()` interface for all optimizers.

---

# Examples

The `examples/` folder contains runnable scripts and short guides demonstrating common workflows. Examples are split by task (classification / regression) and by metric type (sklearn vs custom metric).

Each example is intentionally small and executable; they demonstrate how to:

- Use built-in model configs (`RandomForestConfig`, `SVCConfig`, etc.)
- Provide a custom `SearchSpace` and `custom_metric` callable
- Run an optimizer (GA, PSO, Bayesian) and fit the final model

Run examples locally:

```bash
python examples/classification_examples.py
python examples/regression_examples.py
```

Example 1 — Automatic setup

Use a model config to build the search space and wrapper automatically:

```python
from optiflowx.models.configs import RandomForestConfig
from optiflowx.optimizers import BayesianOptimizer

cfg = RandomForestConfig()
wrapper = cfg.get_wrapper(task_type="classification")
opt = BayesianOptimizer(search_space=cfg.build_search_space(), metric="accuracy", model_class=wrapper.model_class, X=X, y=y)
best_params, best_score = opt.run(max_iters=10)
```

Example 2 — Custom search space and metric

Define a `SearchSpace` and a `custom_metric` callable to integrate bespoke scoring logic:

```python
from optiflowx.core import SearchSpace
from optiflowx.optimizers import GeneticOptimizer

space = SearchSpace()
space.add("n_estimators", "discrete", [20, 200])
space.add("max_depth", "discrete", [2, 20])

def my_metric(y_true, y_pred):
    # return a higher-is-better numeric score
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average='macro'))

opt = GeneticOptimizer(search_space=space, custom_metric=my_metric, model_class=MyClassifier, X=X_train, y=y_train, population=12)
best_params, best_score = opt.run(max_iters=8)
```

Example 3 — Deep learning with PyTorch

Wrap PyTorch models in a `ModelConfig` (see `examples/` for a pattern). Tuning typically explores learning rate, batch size and architecture choices — ensure your config's `get_wrapper()` handles device placement and training loops.

Notes

- For CI-friendly quick runs set `EXAMPLES_FAST_MODE=1` and `EXAMPLES_MAX_ITERS=3`.
- If your `custom_metric` is a nested function or closure, install `dill` to enable multiprocessing serialization.

## Classification example (quick)

This short snippet demonstrates using a built-in model config and a Bayesian optimizer for a classification task.

```python
from sklearn.datasets import load_iris
from optiflowx.models.configs import RandomForestConfig
from optiflowx.optimizers import BayesianOptimizer

X, y = load_iris(return_X_y=True)
cfg = RandomForestConfig()
wrapper = cfg.get_wrapper(task_type='classification')
opt = BayesianOptimizer(
    search_space=cfg.build_search_space(),
    metric='accuracy',
    model_class=wrapper.model_class,
    X=X, y=y,
)
best_params, best_score = opt.run(max_iters=10)
print('Best score:', best_score)
```

## Regression example (quick)

This quick regression example uses a `RandomForestConfig` wrapper with `task_type='regression'` and a negative RMSE metric (higher-is-better convention).

```python
from sklearn.datasets import load_boston
from optiflowx.models.configs import RandomForestConfig
from optiflowx.optimizers import GeneticOptimizer
from sklearn.metrics import mean_squared_error

X, y = load_boston(return_X_y=True)
cfg = RandomForestConfig()
wrapper = cfg.get_wrapper(task_type='regression')

def neg_rmse(y_true, y_pred):
    return -float(mean_squared_error(y_true, y_pred, squared=False))

opt = GeneticOptimizer(
    search_space=cfg.build_search_space(),
    custom_metric=neg_rmse,
    model_class=wrapper.model_class,
    X=X, y=y,
    population=12,
)
best_params, best_score = opt.run(max_iters=8)
print('Best score (neg rmse):', best_score)
```
