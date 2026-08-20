import numpy as np
import pytest

from stochx.timeseries import (
    KalmanFilterResult,
    LinearStateSpace,
    kalman_filter,
    kalman_innovation_diagnostics,
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


def test_innovation_diagnostics_matches_filter_semantics():
    y = np.array([-1.0, -0.5, np.nan, 0.5, 1.0])
    filtered = kalman_filter(y, _model())
    result = kalman_innovation_diagnostics(filtered)

    assert isinstance(filtered, KalmanFilterResult)
    assert result.nobs == 5
    assert result.n_obs == 1
    assert result.effective_nobs[0] == 4
    assert result.missing_observations[0] == 1
    assert result.overall_effective_nobs == 4
    assert result.overall_missing_observations == 1
    assert result.numerically_stable
    np.testing.assert_allclose(
        result.standardized_innovations[np.isfinite(filtered.innovations)],
        filtered.innovations[np.isfinite(filtered.innovations)]
        / np.sqrt(filtered.innovation_cov[np.isfinite(filtered.innovations), 0, 0]),
        rtol=0,
        atol=1e-12,
    )


def test_multivariate_dimensions_are_summarized_independently():
    model = LinearStateSpace(
        transition=np.eye(2),
        design=np.eye(2),
        state_cov=np.eye(2) * 0.1,
        observation_cov=np.eye(2),
        initial_state=np.zeros(2),
        initial_cov=np.eye(2),
    )
    y = np.array([[1.0, 2.0], [np.nan, 3.0], [np.nan, np.nan], [4.0, np.nan]])
    filtered = kalman_filter(y, model)
    result = kalman_innovation_diagnostics(filtered)

    np.testing.assert_array_equal(result.effective_nobs, [2, 2])
    np.testing.assert_array_equal(result.missing_observations, [2, 2])
    assert result.overall_effective_nobs == 4
    assert result.overall_missing_observations == 4
    assert result.standardized_innovations.shape == y.shape
    assert np.isnan(result.standardized_innovations[2]).all()


def test_summary_statistics_are_correct_and_immutable():
    filtered = kalman_filter(np.array([1.0, 2.0, 3.0]), _model())
    result = kalman_innovation_diagnostics(filtered)
    x = filtered.innovations[:, 0]

    np.testing.assert_allclose(result.mean[0], np.mean(x), rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.std[0], np.std(x, ddof=1), rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.rmse[0], np.sqrt(np.mean(x ** 2)), rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.mae[0], np.mean(np.abs(x)), rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.max_abs[0], np.max(np.abs(x)), rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.overall_rmse, result.rmse[0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.overall_mae, result.mae[0], rtol=0, atol=1e-12)
    assert result.innovations.flags.writeable is False
    assert result.standardized_innovations.flags.writeable is False


def test_invalid_filter_result_and_variance_floor_are_rejected():
    with pytest.raises(TypeError, match="filter_result must be a KalmanFilterResult"):
        kalman_innovation_diagnostics(object())

    filtered = kalman_filter(np.array([1.0, 2.0, 3.0]), _model())
    with pytest.raises(ValueError, match="variance_floor must be finite and strictly positive"):
        kalman_innovation_diagnostics(filtered, variance_floor=0.0)
