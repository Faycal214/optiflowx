import numpy as np
import pandas as pd
import pytest

from stochx.timeseries import (
    KalmanFilterResult,
    LinearStateSpace,
    kalman_filter,
    kalman_forecast,
)


def _model():
    return LinearStateSpace(
        transition=np.array([[1.0]]),
        design=np.array([[1.0]]),
        state_cov=np.array([[0.0]]),
        observation_cov=np.array([[1.0]]),
        initial_state=np.array([0.0]),
        initial_cov=np.array([[1.0]]),
    )


def test_scalar_forecast_is_deterministic_and_has_prediction_intervals():
    result = kalman_forecast([1.0, 2.0, 3.0], _model(), steps=3, alpha=0.10)
    assert isinstance(result.filter_result, KalmanFilterResult)
    assert result.horizon == 3
    assert result.alpha == 0.10
    np.testing.assert_allclose(result.forecast[:, 0], [2.0, 2.0, 2.0], rtol=0, atol=1e-12)
    assert result.forecast_cov.shape == (3, 1, 1)
    assert result.standard_error.shape == (3, 1)
    assert np.all(result.lower < result.forecast)
    assert np.all(result.upper > result.forecast)
    assert result.forecast.flags.writeable is False
    assert result.forecast_cov.flags.writeable is False


def test_forecast_reuses_supplied_filter_result_without_changing_stage10_result():
    model = _model()
    observations = np.array([1.0, np.nan, 3.0])
    filtered = kalman_filter(observations, model)
    before = filtered.filtered_state.copy()
    result = kalman_forecast(observations, model, steps=2, filter_result=filtered)
    assert result.filter_result is filtered
    np.testing.assert_array_equal(filtered.filtered_state, before)
    assert filtered.effective_nobs == 2
    assert filtered.missing_observations == 1


def test_datetime_series_gets_future_datetime_index():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    series = pd.Series([1.0, 1.2, 1.1, 1.3], index=dates)
    result = kalman_forecast(series, _model(), steps=3)
    assert isinstance(result.index, pd.DatetimeIndex)
    assert list(result.index) == list(pd.date_range("2024-01-05", periods=3, freq="D"))


def test_multivariate_forecast_preserves_dimensions_and_covariance_symmetry():
    model = LinearStateSpace(
        transition=np.eye(2),
        design=np.eye(2),
        state_cov=np.eye(2) * 0.1,
        observation_cov=np.eye(2),
        initial_state=np.zeros(2),
        initial_cov=np.eye(2),
    )
    result = kalman_forecast([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]], model, steps=4)
    assert result.forecast.shape == (4, 2)
    assert result.forecast_cov.shape == (4, 2, 2)
    np.testing.assert_allclose(
        result.forecast_cov,
        np.swapaxes(result.forecast_cov, -1, -2),
        rtol=1e-12,
        atol=1e-12,
    )
    assert result.lower.shape == result.upper.shape == result.forecast.shape


def test_invalid_forecast_configuration_is_rejected():
    with pytest.raises(ValueError, match="steps must be a positive integer"):
        kalman_forecast([1.0, 2.0], _model(), steps=0)
    with pytest.raises(ValueError, match="alpha must lie strictly between 0 and 1"):
        kalman_forecast([1.0, 2.0], _model(), steps=2, alpha=1.0)


def test_prediction_interval_matches_normal_quantile():
    result = kalman_forecast([1.0, 2.0, 3.0], _model(), steps=1, alpha=0.05)
    z = 1.959963984540054
    expected = result.forecast[0, 0] + z * result.standard_error[0, 0]
    np.testing.assert_allclose(result.upper[0, 0], expected, rtol=0, atol=1e-12)
