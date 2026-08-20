import numpy as np
import pytest

from stochx.timeseries import (
    LinearStateSpace,
    StateSpaceAdequacyResult,
    kalman_filter,
    kalman_innovation_diagnostics,
    state_space_adequacy,
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


def test_state_space_adequacy_returns_one_result_per_dimension():
    rng = np.random.default_rng(123)
    filtered = kalman_filter(rng.normal(size=80), _model())
    diagnostics = kalman_innovation_diagnostics(filtered)
    result = state_space_adequacy(diagnostics, lags=8)

    assert isinstance(result, StateSpaceAdequacyResult)
    assert result.dimensions == 1
    assert len(result.whiteness) == 1
    assert len(result.normality) == 1
    assert len(result.mean_zero) == 1
    assert result.lags == 8
    assert 0 < result.alpha < 1


def test_missing_innovations_are_ignored_consistently():
    y = np.array([0.0, 0.4, np.nan, -0.3, 0.2, np.nan, 0.5, -0.1, 0.2])
    filtered = kalman_filter(y, _model())
    diagnostics = kalman_innovation_diagnostics(filtered)
    result = state_space_adequacy(diagnostics, lags=2)

    assert result.whiteness[0].pvalue == result.whiteness[0].pvalue
    assert result.mean_zero[0].pvalue == result.mean_zero[0].pvalue
    assert result.normality[0].pvalue == result.normality[0].pvalue


def test_short_samples_produce_non_decisive_nan_tests():
    filtered = kalman_filter(np.array([1.0, 2.0, 3.0, 4.0]), _model())
    diagnostics = kalman_innovation_diagnostics(filtered)
    result = state_space_adequacy(diagnostics, lags=5)

    assert np.isnan(result.whiteness[0].pvalue)


def test_invalid_arguments_are_rejected():
    filtered = kalman_filter(np.array([1.0, 2.0, 3.0]), _model())
    diagnostics = kalman_innovation_diagnostics(filtered)

    with pytest.raises(ValueError, match="lags must be a positive integer"):
        state_space_adequacy(diagnostics, lags=0)
    with pytest.raises(ValueError, match="alpha must lie strictly between 0 and 1"):
        state_space_adequacy(diagnostics, alpha=1.0)
    with pytest.raises(TypeError, match="diagnostics must be a KalmanInnovationDiagnosticsResult"):
        state_space_adequacy(object())
