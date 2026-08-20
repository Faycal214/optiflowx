import numpy as np
import pytest

from stochx.timeseries import (
    KalmanFilterResult,
    KalmanSmootherResult,
    LinearStateSpace,
    kalman_filter,
    kalman_smoother,
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


def test_rts_matches_independent_scalar_reference_fixture():
    y = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    result = kalman_smoother(y, _model())
    expected_state = np.array([
        [-0.55333459],
        [-0.32818471],
        [-0.01205951],
        [0.29592552],
        [0.49116028],
    ])
    expected_cov = np.array([
        [[0.19644858]],
        [[0.15228997]],
        [[0.14445603]],
        [[0.14822581]],
        [[0.17549534]],
    ])
    np.testing.assert_allclose(result.smoothed_state, expected_state, rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(result.smoothed_cov, expected_cov, rtol=2e-8, atol=2e-8)


def test_smoother_reuses_filter_result_and_preserves_final_filter_state():
    y = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    model = _model()
    filtered = kalman_filter(y, model)
    result = kalman_smoother(y, model, filter_result=filtered)
    assert isinstance(result, KalmanSmootherResult)
    assert isinstance(result.filter_result, KalmanFilterResult)
    assert result.filter_result is filtered
    np.testing.assert_allclose(result.smoothed_state[-1], filtered.filtered_state[-1], rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.smoothed_cov[-1], filtered.filtered_cov[-1], rtol=0, atol=1e-12)
    assert result.smoothed_state.flags.writeable is False
    assert result.smoothed_cov.flags.writeable is False


def test_one_observation_smoothing_equals_filtering():
    y = np.array([2.0])
    model = _model()
    filtered = kalman_filter(y, model)
    result = kalman_smoother(y, model)
    np.testing.assert_allclose(result.smoothed_state, filtered.filtered_state, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.smoothed_cov, filtered.filtered_cov, rtol=0, atol=1e-12)
    assert result.nobs == 1


def test_missing_observations_preserve_rows_and_dimensions():
    model = LinearStateSpace(
        transition=np.eye(2),
        design=np.eye(2),
        state_cov=np.eye(2) * 0.1,
        observation_cov=np.eye(2),
        initial_state=np.zeros(2),
        initial_cov=np.eye(2),
    )
    y = np.array([[1.0, 2.0], [np.nan, 3.0], [np.nan, np.nan], [4.0, 5.0]])
    result = kalman_smoother(y, model)
    assert result.smoothed_state.shape == (4, 2)
    assert result.smoothed_cov.shape == (4, 2, 2)
    assert result.nobs == 4
    assert result.filter_result.effective_nobs == 5
    assert result.filter_result.missing_observations == 3
    np.testing.assert_array_equal(result.filter_result.observed_dimensions[2], [0, 0])
    assert np.all(np.isfinite(result.smoothed_state))
    np.testing.assert_allclose(result.smoothed_cov, np.swapaxes(result.smoothed_cov, -1, -2), rtol=1e-12, atol=1e-12)


def test_smoother_does_not_mutate_inputs():
    model = _model()
    y = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    original_y = y.copy()
    original_transition = model.transition.copy()
    filtered = kalman_filter(y, model)
    original_filtered = filtered.filtered_state.copy()
    kalman_smoother(y, model, filter_result=filtered)
    np.testing.assert_array_equal(y, original_y)
    np.testing.assert_array_equal(model.transition, original_transition)
    np.testing.assert_array_equal(filtered.filtered_state, original_filtered)


def test_incompatible_filter_result_is_rejected():
    model = _model()
    filtered = kalman_filter(np.array([-1.0, 0.0]), model)
    with pytest.raises(ValueError, match="incompatible"):
        kalman_smoother(np.array([-1.0, 0.0, 1.0]), model, filter_result=filtered)
