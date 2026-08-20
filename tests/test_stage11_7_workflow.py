import numpy as np
import pytest

from stochx.timeseries import StateSpaceWorkflowResult, run_local_level_workflow


def test_stage11_7_runs_complete_local_level_pipeline_deterministically():
    y = np.array([0.2, 0.5, 0.1, 0.8, 0.6, 1.0, 0.7, 1.1, 0.9, 1.2])
    first = run_local_level_workflow(
        y,
        initial_level=0.0,
        initial_variance=1.0,
        diagnostic_lags=2,
        alpha=0.10,
        forecast_steps=3,
    )
    second = run_local_level_workflow(
        y,
        initial_level=0.0,
        initial_variance=1.0,
        diagnostic_lags=2,
        alpha=0.10,
        forecast_steps=3,
    )

    assert isinstance(first, StateSpaceWorkflowResult)
    assert first.success
    assert first.filter_result.nobs == len(y)
    assert first.smoother.nobs == len(y)
    assert first.innovation_diagnostics.nobs == len(y)
    assert first.forecast is not None
    assert first.forecast.horizon == 3
    assert first.smoother.filter_result is first.filter_result

    np.testing.assert_allclose(first.estimation.model.transition, second.estimation.model.transition, rtol=0, atol=1e-12)
    np.testing.assert_allclose(first.filter_result.filtered_state, second.filter_result.filtered_state, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(first.smoother.smoothed_state, second.smoother.smoothed_state, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(first.forecast.forecast, second.forecast.forecast, rtol=1e-10, atol=1e-12)


def test_stage11_7_preserves_missing_observation_semantics():
    y = np.array([0.2, np.nan, 0.8, 1.0, np.nan, 1.2, 1.1, 1.3])
    result = run_local_level_workflow(
        y,
        initial_level=0.0,
        initial_variance=1.0,
        diagnostic_lags=1,
    )

    assert result.filter_result.effective_nobs == 6
    assert result.filter_result.missing_observations == 2
    assert result.innovation_diagnostics.overall_effective_nobs == 6
    assert result.innovation_diagnostics.overall_missing_observations == 2
    assert result.forecast is None
    assert result.smoother.smoothed_state.shape == (len(y), 1)


def test_stage11_7_rejects_invalid_workflow_arguments():
    y = [0.2, 0.5, 0.1, 0.8]
    with pytest.raises(ValueError, match="diagnostic_lags must be a positive integer"):
        run_local_level_workflow(y, diagnostic_lags=0)
    with pytest.raises(ValueError, match="alpha must lie strictly between 0 and 1"):
        run_local_level_workflow(y, alpha=1.0)
    with pytest.raises(ValueError, match="forecast_steps must be a non-negative integer"):
        run_local_level_workflow(y, forecast_steps=-1)
