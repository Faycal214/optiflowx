import os
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from optiflowx.models.configs import RandomForestConfig
from optiflowx.core import ParallelExecutor, Candidate


def neg_mse(y_true, y_pred):
    return -float(mean_squared_error(y_true, y_pred))


def test_parallel_executor_uses_custom_metric():
    # small synthetic regression dataset
    X, y = make_regression(n_samples=50, n_features=4, noise=0.1, random_state=0)
    rf_cfg = RandomForestConfig()
    wrapper = rf_cfg.get_wrapper(task_type="regression")

    params = rf_cfg.build_search_space().sample()
    cand = Candidate(params)

    # Direct evaluation via wrapper (single-process) with the custom metric
    direct_score, _ = wrapper.train_and_score(params, X, y, scoring="mse", custom_metric=neg_mse, task_type="regression")

    # ParallelExecutor should accept the custom metric and return a finite score
    executor = ParallelExecutor(num_workers=1)
    results_mse = executor.evaluate([cand], wrapper, X, y, scoring="mse", custom_metric=neg_mse, task_type="regression")
    assert len(results_mse) == 1
    assert results_mse[0].score is not None and results_mse[0].score != float("-inf")

    # Using a different custom metric should change the reported score
    def neg_mae(y_true, y_pred):
        from sklearn.metrics import mean_absolute_error

        return -float(mean_absolute_error(y_true, y_pred))

    results_mae = executor.evaluate([Candidate(params)], wrapper, X, y, scoring="mse", custom_metric=neg_mae, task_type="regression")
    assert len(results_mae) == 1
    assert results_mae[0].score is not None and results_mae[0].score != float("-inf")
    # The two metrics should produce different numeric values
    assert abs(results_mse[0].score - results_mae[0].score) > 1e-6
