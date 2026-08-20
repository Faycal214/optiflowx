import numpy as np
import pytest

from stochx.timeseries import LinearStateSpace, kalman_filter, local_level_filter


def test_all_missing_scalar_row_is_prediction_only_and_zero_likelihood():
    model = LinearStateSpace(
        transition=np.array([[1.0]]),
        design=np.array([[1.0]]),
        state_cov=np.array([[0.2]]),
        observation_cov=np.array([[0.5]]),
        initial_state=np.array([2.0]),
        initial_cov=np.array([[1.0]]),
    )
    result = kalman_filter([np.nan, np.nan], model)

    assert result.nobs == 2
    assert result.effective_nobs == 0
    assert result.missing_observations == 2
    assert result.loglik == 0.0
    np.testing.assert_array_equal(result.filtered_state, result.predicted_state)
    np.testing.assert_allclose(result.filtered_cov, result.predicted_cov)
    assert np.isnan(result.innovations).all()
    np.testing.assert_array_equal(result.observed_dimensions, [[0], [0]])


def test_singular_zero_noise_model_is_stable_and_deterministic():
    model = LinearStateSpace(
        transition=np.array([[1.0]]),
        design=np.array([[1.0]]),
        state_cov=np.array([[0.0]]),
        observation_cov=np.array([[0.0]]),
        initial_state=np.array([1.0]),
        initial_cov=np.array([[0.0]]),
    )
    first = kalman_filter([1.0, 1.0, 1.0], model)
    second = kalman_filter([1.0, 1.0, 1.0], model)

    assert np.isfinite(first.filtered_state).all()
    assert np.isfinite(first.filtered_cov).all()
    assert np.isfinite(first.loglik)
    np.testing.assert_array_equal(first.filtered_state, second.filtered_state)
    np.testing.assert_array_equal(first.filtered_cov, second.filtered_cov)
    assert first.loglik == second.loglik


def test_state_and_observation_intercepts_are_applied_deterministically():
    model = LinearStateSpace(
        transition=np.array([[1.0]]),
        design=np.array([[1.0]]),
        state_cov=np.array([[0.0]]),
        observation_cov=np.array([[1.0]]),
        initial_state=np.array([0.0]),
        initial_cov=np.array([[1.0]]),
        state_intercept=np.array([0.5]),
        observation_intercept=np.array([1.0]),
    )
    result = kalman_filter([1.0, 2.0], model)

    assert result.nobs == 2
    assert np.isfinite(result.filtered_state).all()
    assert np.isfinite(result.innovations).all()


def test_single_observation_preserves_shapes_and_counts():
    result = local_level_filter([3.0], process_variance=0.1, observation_variance=0.2)

    assert result.nobs == 1
    assert result.effective_nobs == 1
    assert result.missing_observations == 0
    assert result.filtered_state.shape == (1, 1)
    assert result.filtered_cov.shape == (1, 1, 1)
    assert result.innovations.shape == (1, 1)


def test_wrong_observation_dimension_is_rejected():
    model = LinearStateSpace(
        transition=np.eye(2),
        design=np.eye(2),
        state_cov=np.eye(2),
        observation_cov=np.eye(2),
        initial_state=np.zeros(2),
        initial_cov=np.eye(2),
    )
    with pytest.raises(ValueError, match="observations must have shape"):
        kalman_filter([1.0, 2.0, 3.0], model)


def test_nonfinite_model_matrices_are_rejected():
    with pytest.raises(ValueError, match="transition must contain only finite values"):
        LinearStateSpace(
            transition=np.array([[np.nan]]),
            design=np.array([[1.0]]),
            state_cov=np.eye(1),
            observation_cov=np.eye(1),
            initial_state=np.zeros(1),
            initial_cov=np.eye(1),
        )


def test_empty_local_level_sample_is_rejected():
    with pytest.raises(ValueError, match="at least one value"):
        local_level_filter([], process_variance=0.1, observation_variance=0.2)


def test_empty_controls_are_accepted_but_nonempty_controls_are_explicitly_unsupported():
    model = LinearStateSpace(
        transition=np.eye(1),
        design=np.eye(1),
        state_cov=np.eye(1),
        observation_cov=np.eye(1),
        initial_state=np.zeros(1),
        initial_cov=np.eye(1),
    )
    result = kalman_filter([1.0, 2.0], model, controls=np.empty((2, 0)))
    assert result.nobs == 2

    with pytest.raises(ValueError, match="controls are not yet supported"):
        kalman_filter([1.0, 2.0], model, controls=np.ones((2, 1)))
