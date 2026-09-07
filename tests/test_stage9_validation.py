import numpy as np
import pytest

from stochx.timeseries import (
    BoxJenkinsValidationResult,
    estimate_box_jenkins_candidates,
    validate_box_jenkins_candidates,
)


def _ar1(phi: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=n)
    values = np.empty(n)
    values[0] = eps[0]
    for t in range(1, n):
        values[t] = phi * values[t - 1] + eps[t]
    return values


def test_stage9_4_validation_records_frozen_residual_correlogram_and_adequacy():
    y = _ar1(0.0, 220, 3)
    estimation = estimate_box_jenkins_candidates(y, ((0, 0, 0), (1, 0, 0)))
    result = validate_box_jenkins_candidates(estimation, lags=8)
    assert isinstance(result, BoxJenkinsValidationResult)
    assert len(result.records) == 2
