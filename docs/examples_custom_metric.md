# Custom Metric and Regression Examples

This document shows short examples for using `custom_metric` callables and the
`task_type` parameter.

## Custom metric

A custom metric must follow this simple signature:

```python
def my_metric(y_true, y_pred) -> float:
    """Return a numeric score; higher is better."""
    # Example: negative mean absolute error so higher is better
    from sklearn.metrics import mean_absolute_error
    return -float(mean_absolute_error(y_true, y_pred))
```

Pass it to optimizers via the `custom_metric` argument. When using
multiprocessing, non-pickleable callables (e.g., nested functions or lambdas)
are serialized using `dill` if available; otherwise the callable must be
pickleable.

## Regression

To run optimization for regression tasks, set `task_type='regression'` when
obtaining the model wrapper or when creating the `OptimizationEngine`.

Example:

```python
from optiflowx.models.configs.random_forest_config import RandomForestConfig

cfg = RandomForestConfig()
wrapper = cfg.get_wrapper(task_type='regression')

# wrapper can now be used with regressors and regression metrics such as 'mse'
```

For more details and examples, see `README.md`.
