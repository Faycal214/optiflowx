import numpy as np
import pytest

from stochx.timeseries import LocalLevelEstimateResult, estimate_local_level


def test_local_level_estimation_is_deterministic_on_frozen_fixture():
    y = np.array([0.2, 0.5, 0.1, 0.8, 0.6, 1.0, 0.7, 1.1, 0.9, 1.2])
    first = estimate_local_level(y, initial_level=0.0, initial_variance=1.0)
    second = estimate_local_level(y, initial_level=0.0, initial_variance=1.0)

    assert isinstance(first, LocalLevelEstimateResult)
    np.testing.assert_allclose(first.process_variance, second.process_variance, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(first.observation_variance, second.observation_variance, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(first.log_likelihood, second.log_likelihood, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(first.aic, -2.0 * first.log_likelihood + 4.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        first.bic,
        -2.0 * first.log_likelihood + 2.0 * np.log(10.0),
        rtol=0,
        atol=1e-12,
    )
    assert first.iterations == second.iterations
    assert first.message == second.message


def test_local_level_estimation_preserves_missing_observation_semantics():
    y = np.array([0.2, np.nan, 0.8, 1.0, np.nan, 1.2, 1.1, 1.3])
    result = estimate_local_level(y, initial_level=0.0, initial_variance=1.0)

    assert result.success
    assert np.isfinite(result.log_likelihood)
    assert result.process_variance >= 0.0
    assert result.observation_variance >= 0.0


def test_local_level_estimation_rejects_invalid_start_and_observations():
    with pytest.raises(ValueError, match="start variances must be finite and strictly positive"):
        estimate_local_level([1.0, 2.0, 3.0], start=(0.0, 1.0))
    with pytest.raises(ValueError, match="observations must not contain infinite values"):
        estimate_local_level([1.0, np.inf, 2.0])
    with pytest.raises(ValueError, match="at least one finite observation"):
        estimate_local_level([np.nan, np.nan])
    with pytest.raises(ValueError, match="at least two observations"):
        estimate_local_level([1.0])


def test_local_level_estimation_result_is_auditable_and_nonnegative():
    result = estimate_local_level([0.2, 0.5, 0.1, 0.8, 0.6], initial_level=0.0, initial_variance=1.0)

    assert result.success
    assert result.iterations >= 0
    assert isinstance(result.message, str)
    assert result.process_variance >= 0.0
    assert result.observation_variance >= 0.0
    assert result.model.state_cov.shape == (1, 1)
    assert result.model.observation_cov.shape == (1, 1)
