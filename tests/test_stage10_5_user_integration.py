from pathlib import Path

import numpy as np

from stochx.timeseries import LinearStateSpace, KalmanFilterResult, kalman_filter, local_level_filter


def test_stage10_public_kalman_workflow_matches_documented_user_boundary():
    local = local_level_filter(
        [1.0, 2.0, 3.0],
        process_variance=0.0,
        observation_variance=1.0,
        initial_level=0.0,
        initial_variance=1.0,
    )
    assert isinstance(local, KalmanFilterResult)
    np.testing.assert_allclose(local.states[:, 0], [0.5, 1.0, 1.5], rtol=0, atol=1e-12)

    model = LinearStateSpace(
        transition=np.eye(2),
        design=np.eye(2),
        state_cov=np.eye(2) * 0.1,
        observation_cov=np.eye(2),
        initial_state=np.zeros(2),
        initial_cov=np.eye(2),
    )
    result = kalman_filter(
        np.array([[1.0, 2.0], [np.nan, 3.0], [np.nan, np.nan], [4.0, 5.0]]),
        model,
    )
    assert result.nobs == 4
    assert result.effective_nobs == 5
    assert result.missing_observations == 3
    np.testing.assert_array_equal(result.observed_dimensions, [[1, 1], [0, 1], [0, 0], [1, 1]])
    with np.testing.assert_raises(ValueError):
        result.filtered_state[0, 0] = 1.0


def test_stage10_public_example_exists_and_uses_documented_path():
    example = Path(__file__).parents[1] / "examples" / "09_state_space_kalman.py"
    assert example.exists()
    text = example.read_text(encoding="utf-8")
    assert "local_level_filter" in text
    assert "LinearStateSpace" in text
    assert "kalman_filter" in text
