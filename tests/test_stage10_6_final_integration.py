import numpy as np

from stochx.timeseries import (
    BoxJenkinsSelectionResult,
    KalmanFilterResult,
    correlogram,
    estimate_box_jenkins_candidates,
    identify_box_jenkins,
    kalman_filter,
    local_level_filter,
    select_box_jenkins_model,
    validate_box_jenkins_candidates,
    LinearStateSpace,
)


def test_stage10_6_public_cross_stage_integration_preserves_frozen_contracts():
    y = np.array([1.0, 1.2, 1.1, 1.35, 1.25, 1.4, 1.3, 1.5, 1.45, 1.55])

    # Stage 8: frozen correlogram projection.
    corr = correlogram(y, nlags=4)
    assert list(corr.table().columns) == [
        "Lag", "AC", "PAC", "Q-Stat", "Prob.", "DF",
        "AC Lower", "AC Upper", "PAC Lower", "PAC Upper",
    ]
    assert corr.nobs == len(y)
    assert corr.nlags == 4

    # Stage 9: public identification -> estimation -> validation -> selection path.
    ident = identify_box_jenkins(y, d=0, nlags=4, max_p=1, max_q=1)
    assert ident.d == 0
    assert ident.candidate_orders
    estimation = estimate_box_jenkins_candidates(y, ident.candidate_orders)
    validation = validate_box_jenkins_candidates(estimation, lags=2, alpha=0.05)
    selection = select_box_jenkins_model(validation)
    assert isinstance(selection, BoxJenkinsSelectionResult)
    assert selection.status in {"selected", "no_eligible_model"}
    if selection.status == "selected":
        assert selection.selected is not None
        assert selection.selected_order is not None

    # Stage 10: public local-level workflow remains deterministic and auditable.
    local = local_level_filter(
        [1.0, 2.0, 3.0],
        process_variance=0.0,
        observation_variance=1.0,
        initial_level=0.0,
        initial_variance=1.0,
    )
    assert isinstance(local, KalmanFilterResult)
    np.testing.assert_allclose(local.filtered_state[:, 0], [0.5, 1.0, 1.5], rtol=0, atol=1e-12)
    assert local.effective_nobs == 3
    assert local.missing_observations == 0
    assert local.filtered_state.flags.writeable is False


def test_stage10_6_general_state_space_preserves_missing_observation_semantics():
    model = LinearStateSpace(
        transition=np.eye(2),
        design=np.eye(2),
        state_cov=np.eye(2) * 0.1,
        observation_cov=np.eye(2),
        initial_state=np.zeros(2),
        initial_cov=np.eye(2),
    )
    observations = np.array([[1.0, 2.0], [np.nan, 3.0], [np.nan, np.nan], [4.0, 5.0]])

    result = kalman_filter(observations, model)

    assert result.nobs == 4
    assert result.effective_nobs == 5
    assert result.missing_observations == 3
    np.testing.assert_array_equal(result.observed_dimensions[2], [0, 0])
    assert np.isnan(result.innovations[2]).all()
    assert result.filtered_cov.flags.writeable is False
