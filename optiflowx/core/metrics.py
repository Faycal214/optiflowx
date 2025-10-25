from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import inspect

METRICS = {
    "accuracy": lambda y_true, y_pred: accuracy_score(y_true, y_pred),
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
    "precision": lambda y_true, y_pred: precision_score(
        y_true, y_pred, average="macro"
    ),
    "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="macro"),
}


def get_metric(metric):
    """Return a metric function for model evaluation.

    This function validates and retrieves a metric callable used to evaluate
    model predictions. It supports both predefined metric names (e.g., "accuracy")
    and user-defined functions that accept `(y_true, y_pred)` as arguments.

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
