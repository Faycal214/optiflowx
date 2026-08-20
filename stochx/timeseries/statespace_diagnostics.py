"""Stage 11.5 innovation diagnostics for Kalman filter results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .statespace import KalmanFilterResult


@dataclass(frozen=True)
class KalmanInnovationDiagnosticsResult:
    """Auditable descriptive diagnostics for Kalman innovations."""

    innovations: np.ndarray
    standardized_innovations: np.ndarray
    innovation_variance: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    rmse: np.ndarray
    mae: np.ndarray
    max_abs: np.ndarray
    effective_nobs: np.ndarray
    missing_observations: np.ndarray
    nobs: int
    n_obs: int
    numerically_stable: bool
    minimum_positive_variance: float

    def __post_init__(self) -> None:
        arrays = (
            "innovations",
            "standardized_innovations",
            "innovation_variance",
            "mean",
            "std",
            "rmse",
            "mae",
            "max_abs",
            "effective_nobs",
            "missing_observations",
        )
        for name in arrays:
            value = np.asarray(getattr(self, name)).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)

        object.__setattr__(self, "nobs", int(self.nobs))
        object.__setattr__(self, "n_obs", int(self.n_obs))
        object.__setattr__(self, "numerically_stable", bool(self.numerically_stable))
        object.__setattr__(self, "minimum_positive_variance", float(self.minimum_positive_variance))

        if self.innovations.shape != (self.nobs, self.n_obs):
            raise ValueError("innovations shape must match nobs and n_obs")
        if self.standardized_innovations.shape != self.innovations.shape:
            raise ValueError("standardized_innovations must match innovations shape")
        for name in ("innovation_variance", "mean", "std", "rmse", "mae", "max_abs"):
            if getattr(self, name).shape != (self.n_obs,):
                raise ValueError(f"{name} must have one value per observation dimension")
        for name in ("effective_nobs", "missing_observations"):
            value = getattr(self, name)
            if value.shape != (self.n_obs,):
                raise ValueError(f"{name} must have one value per observation dimension")
            if np.any(value < 0):
                raise ValueError(f"{name} must be non-negative")
        if self.effective_nobs.shape != self.missing_observations.shape:
            raise ValueError("observation counts must have matching shapes")
        if not np.isfinite(self.minimum_positive_variance) and self.effective_nobs.any():
            raise ValueError("minimum_positive_variance must be finite when observations are available")

    @property
    def overall_effective_nobs(self) -> int:
        """Total number of observed scalar innovations."""
        return int(self.effective_nobs.sum())

    @property
    def overall_missing_observations(self) -> int:
        """Total number of missing scalar observations."""
        return int(self.missing_observations.sum())

    @property
    def overall_rmse(self) -> float:
        """RMSE over all finite innovations."""
        squared = self.innovations[np.isfinite(self.innovations)]
        return float(np.sqrt(np.mean(squared ** 2))) if squared.size else float("nan")

    @property
    def overall_mae(self) -> float:
        """MAE over all finite innovations."""
        absolute = np.abs(self.innovations[np.isfinite(self.innovations)])
        return float(np.mean(absolute)) if absolute.size else float("nan")


def kalman_innovation_diagnostics(
    filter_result: KalmanFilterResult,
    *,
    variance_floor: float = 1e-12,
) -> KalmanInnovationDiagnosticsResult:
    """Summarize raw and standardized Kalman innovations by dimension.

    The function performs descriptive calculations only. It does not apply
    hypothesis-test thresholds or declare a model adequate/inadequate.
    Missing innovations remain ``NaN`` and do not contribute to the summary
    statistics. Standardization uses ``sqrt(F_t)`` from the filter's innovation
    covariance matrices and therefore preserves the filter's missingness
    semantics.
    """
    if not isinstance(filter_result, KalmanFilterResult):
        raise TypeError("filter_result must be a KalmanFilterResult")
    floor = float(variance_floor)
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("variance_floor must be finite and strictly positive")

    innovations = np.asarray(filter_result.innovations, dtype=float)
    innovation_cov = np.asarray(filter_result.innovation_cov, dtype=float)
    if innovations.ndim != 2:
        raise ValueError("filter_result innovations must be two-dimensional")
    if innovation_cov.shape != (
        filter_result.nobs,
        innovations.shape[1],
        innovations.shape[1],
    ):
        raise ValueError("filter_result innovation_covariance has incompatible shape")

    nobs, n_obs = innovations.shape
    standardized = np.full_like(innovations, np.nan, dtype=float)
    innovation_variance = np.full(n_obs, np.nan, dtype=float)
    mean = np.full(n_obs, np.nan, dtype=float)
    std = np.full(n_obs, np.nan, dtype=float)
    rmse = np.full(n_obs, np.nan, dtype=float)
    mae = np.full(n_obs, np.nan, dtype=float)
    max_abs = np.full(n_obs, np.nan, dtype=float)
    effective_nobs = np.zeros(n_obs, dtype=int)
    missing_observations = np.zeros(n_obs, dtype=int)

    positive_variances: list[float] = []
    numerically_stable = True

    for j in range(n_obs):
        raw = innovations[:, j]
        observed = np.isfinite(raw)
        effective_nobs[j] = int(observed.sum())
        missing_observations[j] = nobs - effective_nobs[j]

        if not observed.any():
            continue

        values = raw[observed]
        finite_vars = innovation_cov[observed, j, j]
        if not np.isfinite(finite_vars).all():
            numerically_stable = False
            continue
        if np.any(finite_vars <= 0):
            numerically_stable = False
        usable = finite_vars > floor
        if not np.any(usable):
            numerically_stable = False
            continue

        positive_variances.extend(finite_vars[usable].tolist())
        z = np.full(values.shape, np.nan, dtype=float)
        z[usable] = values[usable] / np.sqrt(finite_vars[usable])
        standardized[observed, j] = z

        mean[j] = float(np.mean(values))
        std[j] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        rmse[j] = float(np.sqrt(np.mean(values ** 2)))
        mae[j] = float(np.mean(np.abs(values)))
        max_abs[j] = float(np.max(np.abs(values)))
        innovation_variance[j] = float(np.mean(finite_vars))

    minimum_positive_variance = min(positive_variances) if positive_variances else float("nan")
    if positive_variances and minimum_positive_variance <= floor:
        numerically_stable = False

    return KalmanInnovationDiagnosticsResult(
        innovations=innovations,
        standardized_innovations=standardized,
        innovation_variance=innovation_variance,
        mean=mean,
        std=std,
        rmse=rmse,
        mae=mae,
        max_abs=max_abs,
        effective_nobs=effective_nobs,
        missing_observations=missing_observations,
        nobs=nobs,
        n_obs=n_obs,
        numerically_stable=numerically_stable,
        minimum_positive_variance=minimum_positive_variance,
    )
