# Custom metrics and regression examples

OptiFlowX allows you to supply custom metric callables for specialized evaluation. The library treats all metrics as "higher is better"; for error metrics (MSE, RMSE, MAE) consider returning the negative error so optimizers maximize performance.

## Custom metric signature

Your metric should accept `(y_true, y_pred)` and return a numeric scalar:

```python
def my_metric(y_true, y_pred) -> float:
    """Return a numeric score; higher is better."""
    from sklearn.metrics import mean_absolute_error
    return -float(mean_absolute_error(y_true, y_pred))
```

Pass the callable to optimizers via `custom_metric=` or set it on `ModelWrapper.custom_metric`.

### Multiprocessing & serialization

When running parallel evaluations via `ParallelExecutor`, OptiFlowX attempts to serialize the provided callable using `pickle`. If `pickle` is not able to serialize the function (for example, nested functions or lambda closures), and `dill` is installed, the executor will use `dill` automatically.

If neither `pickle` nor `dill` can serialize the metric and multiple workers are requested, the executor will raise a clear error. For single-worker runs the executor falls back to sequential evaluation.

Install `dill` if you plan to use nested/custom closures with multiprocessing:

```bash
pip install dill
```

## Regression workflows

When working with regression tasks, set `task_type='regression'` when getting a model wrapper or when creating an `OptimizationEngine`:

```python
from optiflowx.models.configs.random_forest_config import RandomForestConfig

cfg = RandomForestConfig()
wrapper = cfg.get_wrapper(task_type='regression')

# then use wrapper with regressors and regression-aware metrics
```

Example: custom negative RMSE metric

```python
from sklearn.metrics import mean_squared_error

def neg_rmse(y_true, y_pred):
    return -float(mean_squared_error(y_true, y_pred, squared=False))

# Use as custom_metric in any optimizer
```

For more applied examples see the `examples/` directory and `examples.md`.
