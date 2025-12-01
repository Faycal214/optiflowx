import inspect
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


METRICS = {
    # Classification (higher is better)
    "accuracy": lambda y_true, y_pred: float(accuracy_score(y_true, y_pred)),
    "f1": lambda y_true, y_pred: float(
        f1_score(y_true, y_pred, average="macro")
    ),
    "precision": lambda y_true, y_pred: float(
        precision_score(y_true, y_pred, average="macro")
    ),
    "recall": lambda y_true, y_pred: float(
        recall_score(y_true, y_pred, average="macro")
    ),
    # Regression: by convention we return values where higher is better.
    # For error-based metrics we negate the value so the optimizer can still maximize.
    "mse": lambda y_true, y_pred: -float(mean_squared_error(y_true, y_pred)),
    "rmse": lambda y_true, y_pred: -float(
        mean_squared_error(y_true, y_pred, squared=False)
    ),
    "mae": lambda y_true, y_pred: -float(mean_absolute_error(y_true, y_pred)),
    "r2": lambda y_true, y_pred: float(r2_score(y_true, y_pred)),
}


def get_metric(metric):
    """Return a metric function for model evaluation.

    Supports predefined metric names and user-provided callables that accept
    `(y_true, y_pred)` and return a numeric score. For regression error metrics
    (MSE/MAE/RMSE) the returned value is negated so the optimization framework
    can consistently maximize the score.

    Args:
        metric (str | Callable): Metric name or a custom callable function.

    Returns:
        Callable: A callable that computes a float score given `(y_true, y_pred)`.

    Raises:
        ValueError: If a custom metric does not accept two arguments or if the
            metric name is not recognized.
        TypeError: If `metric` is neither a string nor a callable.
    """
    if callable(metric):
        sig = inspect.signature(metric)
        if len(sig.parameters) >= 2:
            return metric
        raise ValueError("Custom metric must accept at least (y_true, y_pred).")
    if isinstance(metric, str):
        if metric in METRICS:
            return METRICS[metric]
        raise ValueError(f"Unknown metric: {metric}")
    raise TypeError("metric must be str or callable")
