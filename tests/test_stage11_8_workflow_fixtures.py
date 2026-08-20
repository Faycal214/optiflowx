import numpy as np
import pytest

from stochx.timeseries import (
    KalmanSmootherResult,
    StateSpaceWorkflowResult,
    run_local_level_workflow,
)


def _fixture():
    return np.array([0.2, 0.5, 0.1, 0.8, 0.6, 1.0, 0.7, 1.1, 0.9, 1.2], dtype=float)


def test_full_workflow_is_deterministic_and_auditable():
    first = run_local_level_workflow(_fixture(), diagnostic_lags=3, alpha=0.05, forecast_steps=3)
    second = run_local_level_workflow(_fixture(), diagnostic_lags=3, alpha=0.05, forecast_steps=3)

    assert isinstance(first, StateSpaceWorkflowResult)
    assert isinstance(first.smoother, KalmanSmootherResult)
    assert first.success
    assert first.smoother.filter_result is first.filter_result
    assert first.filter_result.nobs == len(_fixture())
    assert first.forecast is not None

    np.testing.assert_allclose(first.estimation.process_variance, second.estimation.process_variance, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(first.estimation.observation_variance, second.estimation.observation_variance, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(first.smoother.smoothed_state, second.smoother.smoothed_state, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(first.forecast.forecast, second.forecast.forecast, rtol=1e-10, atol=1e-12)


def test_workflow_preserves_missingness_without_losing_time_rows():
    y = np.array([0.2, np.nan, 0.8, np.nan, 1.1, 1.3], dtype=float)
    result = run_local_level_workflow(y, diagnostic_lags=2, forecast_steps=0)

    assert result.filter_result.nobs == len(y)
    assert result.filter_result.effective_nobs == 4
    assert result.filter_result.missing_observations == 2
    assert result.innovation_diagnostics.overall_effective_nobs == 4
    assert result.innovation_diagnostics.overall_missing_observations == 2
    assert result.smoother.nobs == len(y)
    assert np.isnan(result.innovation_diagnostics.innovations[1, 0])
    assert result.forecast is None


def test_workflow_forecast_invariants_and_invalid_controls():
    result = run_local_level_workflow(_fixture(), diagnostic_lags=2, forecast_steps=4, alpha=0.10)
    forecast = result.forecast
    assert forecast is not None
    assert forecast.horizon == 4
    assert forecast.forecast.shape == (4, 1)
    assert forecast.forecast_cov.shape == (4, 1, 1)
    assert np.all(forecast.lower[:, 0] <= forecast.forecast[:, 0])
    assert np.all(forecast.forecast[:, 0] <= forecast.upper[:, 0])
    assert forecast.forecast.flags.writeable is False

    with pytest.raises(ValueError, match="forecast_steps must be a non-negative integer"):
        run_local_level_workflow(_fixture(), forecast_steps=-1)
    with pytest.raises(ValueError, match="diagnostic_lags must be a positive integer"):
        run_local_level_workflow(_fixture(), diagnostic_lags=0)
    with pytest.raises(ValueError, match="alpha must lie strictly between 0 and 1"):
        run_local_level_workflow(_fixture(), alpha=1.0)


def test_all_missing_workflow_fails_explicitly_at_estimation_boundary():
    y = np.array([np.nan, np.nan, np.nan], dtype=float)
    with pytest.raises(ValueError, match="at least one finite observation"):
        run_local_level_workflow(y)
