import pytest

from sklearn.datasets import make_classification

from optiflowx.core import ParallelExecutor, Candidate
from optiflowx.models.configs import RandomForestConfig


def test_custom_metric_multiprocessing_serialization():
    """Ensure that a non-top-level custom metric can be serialized using dill
    and that `ParallelExecutor` can evaluate candidates with it. If dill is
    not available, skip this test.
    """
    try:
        import dill  # type: ignore
    except Exception:
        pytest.skip("dill not installed; skipping multiprocessing serialization test")

    X, y = make_classification(n_samples=80, n_features=8, n_informative=6, random_state=1)
    cfg = RandomForestConfig()
    wrapper = cfg.get_wrapper(task_type="classification")

    params = {
        "n_estimators": 8,
        "max_depth": 6,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "bootstrap": True,
        "max_features": "sqrt",
    }

    # nested function -> typically not pickleable by pickle, but dill can handle it
    def nested_metric(y_true, y_pred):
        # simple accuracy-like metric
        return float((y_true == y_pred).mean())

    candidates = [Candidate(params)]
    executor = ParallelExecutor(num_workers=2)
    results = executor.evaluate(candidates, wrapper, X, y, scoring="accuracy", custom_metric=nested_metric, task_type="classification")
    assert len(results) == 1
    res = results[0]
    assert res.score is not None
    assert res.score > float("-inf")
