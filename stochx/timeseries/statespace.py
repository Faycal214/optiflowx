"""Stage 10.2 linear-Gaussian state-space and Kalman filtering core.

The implementation deliberately stays small and deterministic: it provides a
validated linear state-space model and a numerically stable Kalman filter with
explicit missing-observation handling. It does not alter the Stage 8
correlogram or Stage 9 Box-Jenkins APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


_EPS = np.finfo(float).eps


def _as_2d_matrix(name: str, value: np.ndarray | Iterable[float], rows: int | None = None, cols: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D numeric array")
    if rows is not None and arr.shape[0] != rows:
        raise ValueError(f"{name} must have {rows} rows")
    if cols is not None and arr.shape[1] != cols:
        raise ValueError(f"{name} must have {cols} columns")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return arr.copy()


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _stabilize_covariance(covariance: np.ndarray) -> np.ndarray:
    """Return a deterministic positive-definite working covariance."""
    covariance = _symmetrize(covariance)
    try:
        np.linalg.cholesky(covariance)
        return covariance
    except np.linalg.LinAlgError:
        scale = max(1.0, float(np.trace(covariance)))
        jitter = max(_EPS, 1e-12 * scale)
        stabilized = covariance + jitter * np.eye(covariance.shape[0])
        try:
            np.linalg.cholesky(stabilized)
        except np.linalg.LinAlgError as exc:
            raise ValueError("innovation covariance is numerically singular and could not be stabilized") from exc
        return stabilized


@dataclass(frozen=True)
class LinearStateSpace:
    """Validated time-invariant linear-Gaussian state-space model.

    The model follows

    ``x_t = transition @ x_(t-1) + state_noise``

    ``y_t = design @ x_t + observation_noise``

    with zero-mean Gaussian noises of covariance ``state_cov`` and
    ``observation_cov``. Optional intercepts are supplied as vectors through
    ``state_intercept`` and ``observation_intercept``.
    """

    transition: np.ndarray
    design: np.ndarray
    state_cov: np.ndarray
    observation_cov: np.ndarray
    initial_state: np.ndarray
    initial_cov: np.ndarray
    state_intercept: np.ndarray | None = None
    observation_intercept: np.ndarray | None = None

    def __post_init__(self) -> None:
        transition = _as_2d_matrix("transition", self.transition)
        if transition.shape[0] != transition.shape[1]:
            raise ValueError("transition must be square")
        n_state = transition.shape[0]
        design = _as_2d_matrix("design", self.design, cols=n_state)
        n_obs = design.shape[0]
        state_cov = _as_2d_matrix("state_cov", self.state_cov, rows=n_state, cols=n_state)
        observation_cov = _as_2d_matrix("observation_cov", self.observation_cov, rows=n_obs, cols=n_obs)
        initial_state = np.asarray(self.initial_state, dtype=float).reshape(-1)
        initial_cov = _as_2d_matrix("initial_cov", self.initial_cov, rows=n_state, cols=n_state)
        if initial_state.size != n_state:
            raise ValueError("initial_state has the wrong dimension")
        state_intercept = np.zeros(n_state, dtype=float) if self.state_intercept is None else np.asarray(self.state_intercept, dtype=float).reshape(-1)
        observation_intercept = np.zeros(n_obs, dtype=float) if self.observation_intercept is None else np.asarray(self.observation_intercept, dtype=float).reshape(-1)
        if state_intercept.size != n_state:
            raise ValueError("state_intercept has the wrong dimension")
        if observation_intercept.size != n_obs:
            raise ValueError("observation_intercept has the wrong dimension")
        if not np.isfinite(initial_state).all() or not np.isfinite(state_intercept).all() or not np.isfinite(observation_intercept).all():
            raise ValueError("initial_state and intercepts must be finite")
        state_cov = _symmetrize(state_cov)
        observation_cov = _symmetrize(observation_cov)
        initial_cov = _symmetrize(initial_cov)
        for name, cov in (("state_cov", state_cov), ("observation_cov", observation_cov), ("initial_cov", initial_cov)):
            eig = np.linalg.eigvalsh(cov)
            if eig.min() < -1e-12:
                raise ValueError(f"{name} must be positive semidefinite")
        for name, value in {
            "transition": transition,
            "design": design,
            "state_cov": state_cov,
            "observation_cov": observation_cov,
            "initial_state": initial_state,
            "initial_cov": initial_cov,
            "state_intercept": state_intercept,
            "observation_intercept": observation_intercept,
        }.items():
            copied = np.asarray(value, dtype=float).copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)

    @property
    def n_state(self) -> int:
        return self.transition.shape[0]

    @property
    def n_obs(self) -> int:
        return self.design.shape[0]


@dataclass(frozen=True)
class KalmanFilterResult:
    """Auditable Kalman filtering output with immutable numerical arrays."""

    filtered_state: np.ndarray
    predicted_state: np.ndarray
    filtered_cov: np.ndarray
    predicted_cov: np.ndarray
    innovations: np.ndarray
    innovation_cov: np.ndarray
    loglik: float
    nobs: int
    effective_nobs: int
    missing_observations: int
    observed_dimensions: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            "filtered_state",
            "predicted_state",
            "filtered_cov",
            "predicted_cov",
            "innovations",
            "innovation_cov",
            "observed_dimensions",
        )
        for name in arrays:
            value = np.asarray(getattr(self, name), dtype=float if name != "observed_dimensions" else int).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "loglik", float(self.loglik))
        object.__setattr__(self, "nobs", int(self.nobs))
        object.__setattr__(self, "effective_nobs", int(self.effective_nobs))
        object.__setattr__(self, "missing_observations", int(self.missing_observations))
        if self.nobs != self.filtered_state.shape[0]:
            raise ValueError("nobs must match filtered_state length")
        total_measurements = self.nobs * self.innovation_cov.shape[1]
        if self.effective_nobs < 0 or self.effective_nobs > total_measurements:
            raise ValueError("effective_nobs is out of bounds")
        if self.missing_observations < 0 or self.missing_observations > total_measurements:
            raise ValueError("missing_observations is out of bounds")
        if self.effective_nobs + self.missing_observations != total_measurements:
            raise ValueError("effective_nobs plus missing_observations must equal total scalar measurements")

    @property
    def states(self) -> np.ndarray:
        """Alias for filtered state means."""
        return self.filtered_state

    @property
    def log_likelihood(self) -> float:
        """Alias for the Gaussian observation log likelihood."""
        return self.loglik


def _normal_loglik(innovation: np.ndarray, covariance: np.ndarray) -> tuple[float, np.ndarray]:
    covariance = _stabilize_covariance(covariance)
    chol = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(chol, innovation)
    logdet = 2.0 * float(np.log(np.diag(chol)).sum())
    loglik = -0.5 * (innovation.size * np.log(2.0 * np.pi) + logdet + float(whitened @ whitened))
    return loglik, covariance


def kalman_filter(
    observations: np.ndarray | Iterable[float],
    model: LinearStateSpace,
    *,
    controls: np.ndarray | None = None,
) -> KalmanFilterResult:
    """Run the deterministic Kalman filter for a linear-Gaussian model.

    Missing observations are handled per observation dimension: NaN values are
    excluded from that update, while finite dimensions still update the state.
    A row containing only NaNs performs prediction only and contributes zero to
    the Gaussian log likelihood.
    """
    y = np.asarray(observations, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.ndim != 2 or y.shape[1] != model.n_obs:
        raise ValueError(f"observations must have shape (n, {model.n_obs})")
    if np.isinf(y).any():
        raise ValueError("observations must not contain infinite values")
    n = y.shape[0]
    if n < 1:
        raise ValueError("observations must contain at least one row")

    if controls is not None:
        u = np.asarray(controls, dtype=float)
        if u.ndim == 1:
            u = u.reshape(n, -1)
        if u.ndim != 2 or u.shape[0] != n:
            raise ValueError("controls must have one row per observation")
        if not np.isfinite(u).all():
            raise ValueError("controls must contain only finite values")
        if u.shape[1] != 0:
            raise ValueError("controls are not yet supported by the Stage 10.2 core")

    filtered_state = np.empty((n, model.n_state), dtype=float)
    predicted_state = np.empty_like(filtered_state)
    filtered_cov = np.empty((n, model.n_state, model.n_state), dtype=float)
    predicted_cov = np.empty_like(filtered_cov)
    innovations = np.full((n, model.n_obs), np.nan, dtype=float)
    innovation_cov = np.full((n, model.n_obs, model.n_obs), np.nan, dtype=float)
    observed_dimensions = np.zeros((n, model.n_obs), dtype=int)

    state = model.initial_state.copy()
    covariance = model.initial_cov.copy()
    loglik = 0.0
    effective_nobs = 0

    for t in range(n):
        state = model.transition @ state + model.state_intercept
        covariance = _symmetrize(model.transition @ covariance @ model.transition.T + model.state_cov)
        predicted_state[t] = state
        predicted_cov[t] = covariance

        observed = np.isfinite(y[t])
        observed_dimensions[t] = observed.astype(int)
        if not observed.any():
            filtered_state[t] = state
            filtered_cov[t] = covariance
            continue

        idx = np.flatnonzero(observed)
        H = model.design[idx]
        R = model.observation_cov[np.ix_(idx, idx)]
        measurement = y[t, idx] - model.observation_intercept[idx]
        innovation = measurement - H @ state
        S = _stabilize_covariance(H @ covariance @ H.T + R)
        gain = np.linalg.solve(S, H @ covariance).T
        updated_state = state + gain @ innovation
        identity = np.eye(model.n_state)
        joseph_left = identity - gain @ H
        updated_cov = joseph_left @ covariance @ joseph_left.T + gain @ R @ gain.T
        updated_cov = _symmetrize(updated_cov)

        innovations[t, idx] = innovation
        innovation_cov[t][np.ix_(idx, idx)] = S
        increment, _ = _normal_loglik(innovation, S)
        loglik += increment
        effective_nobs += idx.size
        state = updated_state
        covariance = updated_cov
        filtered_state[t] = state
        filtered_cov[t] = covariance

    total_measurements = n * model.n_obs
    missing_observations = total_measurements - effective_nobs
    return KalmanFilterResult(
        filtered_state=filtered_state,
        predicted_state=predicted_state,
        filtered_cov=filtered_cov,
        predicted_cov=predicted_cov,
        innovations=innovations,
        innovation_cov=innovation_cov,
        loglik=loglik,
        nobs=n,
        effective_nobs=effective_nobs,
        missing_observations=missing_observations,
        observed_dimensions=observed_dimensions,
    )


def local_level_filter(
    observations: np.ndarray | Iterable[float],
    *,
    process_variance: float,
    observation_variance: float,
    initial_level: float | None = None,
    initial_variance: float | None = None,
) -> KalmanFilterResult:
    """Convenience filter for the scalar local-level model.

    ``level_t = level_(t-1) + eta_t`` and ``y_t = level_t + eps_t``.
    """
    q = float(process_variance)
    r = float(observation_variance)
    if q < 0 or r < 0 or not np.isfinite(q + r):
        raise ValueError("process_variance and observation_variance must be finite and non-negative")
    values = np.asarray(observations, dtype=float).reshape(-1)
    if values.size < 1:
        raise ValueError("observations must contain at least one value")
    finite = values[np.isfinite(values)]
    if initial_level is None:
        initial_level = float(finite[0]) if finite.size else 0.0
    if initial_variance is None:
        initial_variance = 1e6 if q > 0 else 1.0
    if initial_variance < 0 or not np.isfinite(initial_variance):
        raise ValueError("initial_variance must be finite and non-negative")
    model = LinearStateSpace(
        transition=np.array([[1.0]]),
        design=np.array([[1.0]]),
        state_cov=np.array([[q]]),
        observation_cov=np.array([[r]]),
        initial_state=np.array([initial_level]),
        initial_cov=np.array([[initial_variance]]),
    )
    return kalman_filter(values, model)
