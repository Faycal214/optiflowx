"""Public Stage 10 state-space and Kalman-filter example."""

from __future__ import annotations

import numpy as np

from stochx.timeseries import LinearStateSpace, kalman_filter, local_level_filter


if __name__ == "__main__":
    local = local_level_filter(
        [1.0, 2.0, 3.0],
        process_variance=0.0,
        observation_variance=1.0,
        initial_level=0.0,
        initial_variance=1.0,
    )
    print("Local-level filtered states:", local.states[:, 0])
    print("Local-level log likelihood:", local.log_likelihood)

    model = LinearStateSpace(
        transition=np.eye(2),
        design=np.eye(2),
        state_cov=np.eye(2) * 0.1,
        observation_cov=np.eye(2),
        initial_state=np.zeros(2),
        initial_cov=np.eye(2),
    )
    observations = np.array(
        [[1.0, 2.0], [np.nan, 3.0], [np.nan, np.nan], [4.0, 5.0]],
        dtype=float,
    )
    result = kalman_filter(observations, model)
    print("Filtered states:")
    print(result.filtered_state)
    print("nobs:", result.nobs)
    print("effective_nobs:", result.effective_nobs)
    print("missing_observations:", result.missing_observations)
    print("observed_dimensions:")
    print(result.observed_dimensions)
