import numpy as np
import pytest

from stochx.timeseries import KalmanFilterResult, LinearStateSpace, kalman_filter, local_level_filter


def test_local_level_filter_has_deterministic_scalar_reference_values():
    result = local_level_filter(
        [1.0, 2.0, 3.0],
        process_variance=0.0,
        observation_variance=1.0,
        initial_level=0.0,
        initial_variance=1.0,
    )

    np.testing.assert_allclose(result.predicted_state[:, 0], [0.0, 0.5, 1.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.filtered_state[:, 0], [0.5, 1.0, 1.5], rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.predicted_cov[:, 0, 0], [1.0, 0.5, 1.0 / 3.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.filtered_cov[:, 0, 0], [0.5, 1.0 / 3.0, 0.25], rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.innovations[:, 0], [1.0, 1.5, 2.0], rtol=0, atol=1e-12)
    assert result.nobs == 3
    assert result.effective_nobs == 3
    assert result.missing_observations == 0
    assert np.isfinite(result.loglik)


def test_partial_and_full_missing_observations_are_handled_without_losing_dimensions():
    model = LinearStateSpace(
        transition=np.eye(2),
        design=np.eye(2),
        state_cov=np.eye(2) * 0.1,
        observation_cov=np.eye(2),
        initial_state=np.zeros(2),
        initial_cov=np.eye(2),
    )
    y = np.array([[1.0, 2.0], [np.nan, 3.0], [np.nan, np.nan], [4.0, 5.0]])

    result = kalman_filter(y, model)

    assert result.nobs == 4
    assert result.effective_nobs == 6
    assert result.missing_observations == 1
    np.testing.assert_array_equal(result.observed_dimensions[2], [0, 0])
    assert np.isnan(result.innovations[2]).all()
    assert np.isfinite(result.filtered_state).all()


def test_filtering_is_deterministic_and_result_arrays_are_immutable():
    model = LinearStateSpace(
        transition=np.array([[0.8]]),
        design=np.array([[1.0]]),
        state_cov=np.array([[0.2]]),
        observation_cov=np.array([[0.4]]),
        initial_state=np.array([0.0]),
        initial_cov=np.array([[1.0]]),
    )
    y = np.linspace(-1.0, 1.0, 20)

    first = kalman_filter(y, model)
    second = kalman_filter(y, model)

    np.testing.assert_array_equal(first.filtered_state, second.filtered_state)
    np.testing.assert_array_equal(first.filtered_cov, second.filtered_cov)
    assert first.loglik == second.loglik
    with pytest.raises(ValueError):
        first.filtered_state[0, 0] = 99.0


def test_model_validation_rejects_bad_shapes_and_negative_covariance():
    with pytest.raises(ValueError, match="transition must be square"):
        LinearStateSpace(
            transition=np.ones((1, 2)),
            design=np.ones((1, 1)),
            state_cov=np.eye(1),
            observation_cov=np.eye(1),
            initial_state=np.zeros(1),
            initial_cov=np.eye(1),
        )

    with pytest.raises(ValueError, match="positive semidefinite"):
        LinearStateSpace(
            transition=np.eye(1),
            design=np.ones((1, 1)),
            state_cov=np.array([[-0.1]]),
            observation_cov=np.eye(1),
            initial_state=np.zeros(1),
            initial_cov=np.eye(1),
        )


def test_filter_rejects_infinite_observations_and_empty_samples():
    model = LinearStateSpace(
        transition=np.eye(1),
        design=np.eye(1),
        state_cov=np.eye(1),
        observation_cov=np.eye(1),
        initial_state=np.zeros(1),
        initial_cov=np.eye(1),
    )
    with pytest.raises(ValueError, match="infinite"):
        kalman_filter([1.0, np.inf], model)
    with pytest.raises(ValueError, match="at least one row"):
        kalman_filter([], model)


def test_result_type_is_public_and_auditable():
    result = local_level_filter([1.0, 1.5], process_variance=0.1, observation_variance=0.2)
    assert isinstance(result, KalmanFilterResult)
    assert result.states is result.filtered_state
    assert result.log_likelihood == result.loglik
