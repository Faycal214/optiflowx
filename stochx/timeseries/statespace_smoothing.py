"""RTS smoothing extension for the Stage 10 linear-Gaussian state-space core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .statespace import KalmanFilterResult, LinearStateSpace, kalman_filter


@dataclass(frozen=True)
class KalmanSmootherResult:
    """Auditable Rauch-Tung-Striebel smoothing output."""

    smoothed_state: np.ndarray
    smoothed_cov: np.ndarray
    filter_result: KalmanFilterResult
    nobs: int

    def __post_init__(self) -> None:
        state = np.asarray(self.smoothed_state, dtype=float).copy()
        cov = np.asarray(self.smoothed_cov, dtype=float).copy()
        state.setflags(write=False)
        cov.setflags(write=False)
        object.__setattr__(self, "smoothed_state", state)
        object.__setattr__(self, "smoothed_cov", cov)
        object.__setattr__(self, "nobs", int(self.nobs))
        if not isinstance(self.filter_result, KalmanFilterResult):
            raise TypeError("filter_result must be a KalmanFilterResult")
        if self.nobs != self.filter_result.nobs:
            raise ValueError("nobs must match filter_result.nobs")
        if self.smoothed_state.ndim != 2:
            raise ValueError("smoothed_state must be two-dimensional")
        if self.smoothed_cov.ndim != 3:
            raise ValueError("smoothed_cov must be three-dimensional")
        if self.smoothed_state.shape[0] != self.nobs:
            raise ValueError("smoothed_state length must match nobs")
        if self.smoothed_cov.shape[0] != self.nobs:
            raise ValueError("smoothed_cov length must match nobs")
        if self.smoothed_cov.shape[1:] != (self.smoothed_state.shape[1],) * 2:
            raise ValueError("smoothed_cov state dimensions must match smoothed_state")

    @property
    def states(self) -> np.ndarray:
        """Alias for smoothed state means."""
        return self.smoothed_state

    @property
    def covariances(self) -> np.ndarray:
        """Alias for smoothed covariance matrices."""
        return self.smoothed_cov


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _stable_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    matrix = _symmetrize(np.asarray(matrix, dtype=float))
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        scale = max(1.0, float(np.trace(matrix)))
        jitter = max(np.finfo(float).eps, 1e-12 * scale)
        try:
            return np.linalg.solve(matrix + jitter * np.eye(matrix.shape[0]), rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError("unable to solve the RTS smoothing covariance system") from exc


def _validate_filter_result(
    observations: np.ndarray,
    model: LinearStateSpace,
    filter_result: KalmanFilterResult,
) -> None:
    if not isinstance(filter_result, KalmanFilterResult):
        raise TypeError("filter_result must be a KalmanFilterResult")
    y = np.asarray(observations, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.ndim != 2 or y.shape[1] != model.n_obs:
        raise ValueError(f"observations must have shape (n, {model.n_obs})")
    if y.shape[0] != filter_result.nobs:
        raise ValueError("filter_result is incompatible with observations")
    if np.isinf(y).any():
        raise ValueError("observations must not contain infinite values")
    expected_mask = np.isfinite(y).astype(int)
    if not np.array_equal(expected_mask, filter_result.observed_dimensions):
        raise ValueError("filter_result is incompatible with the observation missingness pattern")
    if filter_result.predicted_state.shape[1] != model.n_state:
        raise ValueError("filter_result is incompatible with model state dimension")
    if filter_result.predicted_cov.shape[1:] != (model.n_state, model.n_state):
        raise ValueError("filter_result is incompatible with model covariance dimension")


def kalman_smoother(
    observations: np.ndarray | Iterable[float],
    model: LinearStateSpace,
    *,
    filter_result: KalmanFilterResult | None = None,
) -> KalmanSmootherResult:
    """Run a deterministic Rauch-Tung-Striebel smoother.

    The forward pass reuses ``filter_result`` when supplied. The backward
    recursion uses the Stage 10 filtered and predicted states/covariances and
    never mutates either the observations or the filter result.
    """
    y = np.asarray(observations, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.ndim != 2 or y.shape[1] != model.n_obs:
        raise ValueError(f"observations must have shape (n, {model.n_obs})")
    if y.shape[0] < 1:
        raise ValueError("observations must contain at least one row")
    if np.isinf(y).any():
        raise ValueError("observations must not contain infinite values")

    if filter_result is None:
        filter_result = kalman_filter(y, model)
    else:
        _validate_filter_result(y, model, filter_result)

    n = filter_result.nobs
    k = model.n_state
    smoothed_state = filter_result.filtered_state.copy()
    smoothed_cov = filter_result.filtered_cov.copy()

    for t in range(n - 2, -1, -1):
        filtered_cov_t = filter_result.filtered_cov[t]
        predicted_cov_next = filter_result.predicted_cov[t + 1]
        transition = model.transition

        # J_t = P_t F' (P_{t+1|t})^{-1}, evaluated by a solve rather than
        # forming an explicit matrix inverse.
        gain = _stable_solve(
            predicted_cov_next,
            (filtered_cov_t @ transition.T).T,
        ).T

        smoothed_state[t] = filter_result.filtered_state[t] + gain @ (
            smoothed_state[t + 1] - filter_result.predicted_state[t + 1]
        )
        smoothed_cov[t] = _symmetrize(
            filtered_cov_t
            + gain @ (smoothed_cov[t + 1] - predicted_cov_next) @ gain.T
        )

    return KalmanSmootherResult(
        smoothed_state=smoothed_state,
        smoothed_cov=smoothed_cov,
        filter_result=filter_result,
        nobs=n,
    )
