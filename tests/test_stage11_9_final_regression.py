import numpy as np

from stochx.timeseries import (
    correlogram,
    identify_box_jenkins,
    estimate_box_jenkins_candidates,
    validate_box_jenkins_candidates,
    select_box_jenkins_model,
    LinearStateSpace,
    kalman_filter,
    kalman_smoother,
    kalman_innovation_diagnostics,
    state_space_adequacy,
    kalman_forecast,
    run_local_level_workflow,
)


def _model():
    return LinearStateSpace(
        transition=np.array([[0.8]]),
        design=np.array([[1.0]]),
        state_cov=np.array([[0.2]]),
        observation_cov=np.array([[0.4]]),
        initial_state=np.array([0.0]),
        initial_cov=np.array([[1.0]]),
    )


def test_stage11_final_cross_stage_regression_path():
    y = np.array([1.0, 1.2, 1.1, 1.35, 1.25, 1.4, 1.3, 1.5, 1.45, 1.55])

    # Stage 8 contract remains operational.
    corr = correlogram(y, nlags=4)
    assert list(corr.table().columns) == [
        "Lag", "AC", "PAC", "Q-Stat", "Prob.", "DF",
        "AC Lower", "AC Upper", "PAC Lower", "PAC Upper",
    ]

    # Stage 9 public path remains operational.
    ident = identify_box_jenkins(y, d=0, nlags=4, max_p=1, max_q=1)
    estimation = estimate_box_jenkins_candidates(y, ident.candidate_orders)
    validation = validate_box_jenkins_candidates(estimation, lags=2, alpha=0.05)
    selection = select_box_jenkins_model(validation)
    assert selection.status in {"selected", "no_eligible_model"}

    # Stage 10 + Stage 11 extension path.
    filtered = kalman_filter(y, _model())
    smoothed = kalman_smoother(y, _model(), filter_result=filtered)
    diagnostics = kalman_innovation_diagnostics(filtered)
    adequacy = state_space_adequacy(diagnostics, lags=2, alpha=0.05)
    forecast = kalman_forecast(y, _model(), steps=3, alpha=0.10, filter_result=filtered)

    assert smoothed.filter_result is filtered
    assert diagnostics.nobs == filtered.nobs
    assert adequacy.dimensions == 1
    assert forecast.filter_result is filtered
    np.testing.assert_array_equal(smoothed.smoothed_state.shape, (len(y), 1))
    np.testing.assert_array_equal(forecast.forecast.shape, (3, 1))
    assert np.all(forecast.lower <= forecast.forecast)
    assert np.all(forecast.forecast <= forecast.upper)


def test_stage11_workflow_is_deterministic_and_reuses_filter_result():
    y = np.array([1.0, 1.2, np.nan, 1.3, 1.4, 1.25, 1.5, 1.55])
    first = run_local_level_workflow(y, diagnostic_lags=2, alpha=0.10, forecast_steps=3)
    second = run_local_level_workflow(y, diagnostic_lags=2, alpha=0.10, forecast_steps=3)

    assert first.filter_result is first.smoother.filter_result
    assert first.filter_result is first.forecast.filter_result
    np.testing.assert_allclose(first.smoother.smoothed_state, second.smoother.smoothed_state, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(first.forecast.forecast, second.forecast.forecast, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(first.forecast.lower, second.forecast.lower, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(first.forecast.upper, second.forecast.upper, rtol=1e-10, atol=1e-12)


def test_stage11_workflow_preserves_missingness_and_shapes():
    y = np.array([1.0, np.nan, 1.4, np.nan, 1.6])
    result = run_local_level_workflow(y, diagnostic_lags=1, forecast_steps=2)

    assert result.filter_result.nobs == 5
    assert result.filter_result.effective_nobs == 3
    assert result.filter_result.missing_observations == 2
    assert result.innovation_diagnostics.overall_effective_nobs == 3
    assert result.forecast.forecast.shape == (2, 1)
    assert result.smoother.smoothed_state.shape == (5, 1)
