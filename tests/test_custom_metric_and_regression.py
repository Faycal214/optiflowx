import pytest
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

from optiflowx.models.configs import RandomForestConfig


def test_regression_with_custom_metric():
    X, y = make_regression(n_samples=120, n_features=6, noise=0.1, random_state=0)
    cfg = RandomForestConfig()
    wrapper = cfg.get_wrapper(task_type="regression")

    params = {
        "n_estimators": 10,
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "bootstrap": True,
        "max_features": "sqrt",
    }

    def neg_mse(y_true, y_pred):
        return -float(mean_squared_error(y_true, y_pred))

    score, model = wrapper.train_and_score(params, X, y, cv=3, custom_metric=neg_mse, task_type="regression")
    assert isinstance(score, float)
    # neg_mse should be <= 0 (since it's -MSE)
    assert score <= 0.0
    assert model is not None
