import numpy as np
import pytest

from stochx.timeseries import LocalLevelEstimateResult, estimate_local_level


def test_local_level_estimation_is_deterministic_and_auditable():
    y = np.array([1.0, 1.2, 0.9, 1.4, 1.1, 1.3, 1.0, 1.5, 1.2, 1.4])
    a = estimate_local_level(y, start=(0.1, 0.9))
    b = estimate_local_level(y, start=(0.1, 0.9))
    assert isinstance(a, LocalLevelEstimateResult)
    np.testing.assert_allclose(a.process_variance, b.process_variance, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a.observation_variance, b.observation_variance, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a.log_likelihood, b.log_likelihood, rtol=0, atol=1e-12)
    assert np.isfinite(a.aic)
    assert np.isfinite(a.bic)
    assert a.iterations >= 0
    assert a.message


def test_local_level_estimation_handles_missing_observations():
    y = np.array([1.0, np.nan, 1.4, 1.2, np.nan, 1.5, 1.1])
    result = estimate_local_level(y)
    assert result.process_variance >= 0
    assert result.observation_variance >= 0
    assert np.isfinite(result.log_likelihood)


def test_local_level_estimation_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="at least two observations"):
        estimate_local_level([1.0])
    with pytest.raises(ValueError, match="infinite"):
        estimate_local_level([1.0, np.inf, 2.0])
    with pytest.raises(ValueError, match="start variances"):
        estimate_local_level([1.0, 1.1], start=(0.0, 1.0))


def test_local_level_estimation_preserves_parameterization():
    result = estimate_local_level([1.0, 1.2, 1.1, 1.3], start=(0.2, 0.8))
    assert result.model.n_state == 1
    assert result.model.n_obs == 1
    np.testing.assert_allclose(result.model.state_cov, [[result.process_variance]])
    np.testing.assert_allclose(result.model.observation_cov, [[result.observation_variance]])
