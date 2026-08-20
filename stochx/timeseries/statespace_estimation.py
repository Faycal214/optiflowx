"""Stage 11.3 Gaussian likelihood estimation for the scalar local-level model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import minimize

from .statespace import LinearStateSpace, local_level_filter


@dataclass(frozen=True)
class LocalLevelEstimateResult:
    """Auditable maximum-likelihood estimate for local-level variances."""

    process_variance: float
    observation_variance: float
    log_likelihood: float
    aic: float
    bic: float
    success: bool
    iterations: int
    message: str
    model: LinearStateSpace

    def __post_init__(self) -> None:
        object.__setattr__(self, "process_variance", float(self.process_variance))
        object.__setattr__(self, "observation_variance", float(self.observation_variance))
        object.__setattr__(self, "log_likelihood", float(self.log_likelihood))
        object.__setattr__(self, "aic", float(self.aic))
        object.__setattr__(self, "bic", float(self.bic))
        object.__setattr__(self, "success", bool(self.success))
        object.__setattr__(self, "iterations", int(self.iterations))
        object.__setattr__(self, "message", str(self.message))
        if self.process_variance < 0 or self.observation_variance < 0:
            raise ValueError("estimated variances must be non-negative")
        if not isinstance(self.model, LinearStateSpace):
            raise TypeError("model must be a LinearStateSpace")


def _prepare(values: np.ndarray | Iterable[float]) -> np.ndarray:
    y = np.asarray(values, dtype=float).reshape(-1)
    if y.size < 2:
        raise ValueError("estimation requires at least two observations")
    if np.isinf(y).any():
        raise ValueError("observations must not contain infinite values")
    if np.isnan(y).all():
        raise ValueError("estimation requires at least one finite observation")
    return y.copy()


def estimate_local_level(
    observations: np.ndarray | Iterable[float],
    *,
    initial_level: float | None = None,
    initial_variance: float | None = None,
    start: tuple[float, float] = (0.1, 0.9),
) -> LocalLevelEstimateResult:
    """Estimate process/observation variances by deterministic Gaussian ML.

    The state transition and observation equations remain those of the frozen
    Stage 10 local-level model. Only the two non-negative variance parameters
    are estimated. Optimization is performed in log-variance space with
    L-BFGS-B, using a fixed starting point and a fixed deterministic objective.
    """
    y = _prepare(observations)
    start_q, start_r = (float(start[0]), float(start[1]))
    if start_q <= 0 or start_r <= 0 or not np.isfinite(start_q + start_r):
        raise ValueError("start variances must be finite and strictly positive")
    if initial_variance is not None and (initial_variance < 0 or not np.isfinite(initial_variance)):
        raise ValueError("initial_variance must be finite and non-negative")
    if initial_level is not None and not np.isfinite(initial_level):
        raise ValueError("initial_level must be finite")

    def objective(log_variances: np.ndarray) -> float:
        q, r = np.exp(log_variances)
        result = local_level_filter(
            y,
            process_variance=float(q),
            observation_variance=float(r),
            initial_level=initial_level,
            initial_variance=initial_variance,
        )
        if not np.isfinite(result.log_likelihood):
            return 1e100
        return -result.log_likelihood

    opt = minimize(
        objective,
        np.log(np.array([start_q, start_r], dtype=float)),
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-10, "maxls": 50},
    )
    q_hat, r_hat = np.exp(opt.x)
    fitted = local_level_filter(
        y,
        process_variance=float(q_hat),
        observation_variance=float(r_hat),
        initial_level=initial_level,
        initial_variance=initial_variance,
    )
    k = 2
    n_eff = fitted.effective_nobs
    llf = fitted.log_likelihood
    aic = -2.0 * llf + 2.0 * k
    bic = -2.0 * llf + float(k * np.log(max(1, n_eff)))
    model = LinearStateSpace(
        transition=np.array([[1.0]]),
        design=np.array([[1.0]]),
        state_cov=np.array([[float(q_hat)]]),
        observation_cov=np.array([[float(r_hat)]]),
        initial_state=np.array([fitted.filtered_state[0, 0]]),
        initial_cov=np.array([[fitted.filtered_cov[0, 0, 0]]]),
    )
    return LocalLevelEstimateResult(
        process_variance=float(q_hat),
        observation_variance=float(r_hat),
        log_likelihood=llf,
        aic=aic,
        bic=bic,
        success=bool(opt.success),
        iterations=int(getattr(opt, "nit", 0)),
        message=str(opt.message),
        model=model,
    )
