"""Forecasting extension for the Stage 10 linear-Gaussian state-space core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm

from .statespace import KalmanFilterResult, LinearStateSpace, kalman_filter


@dataclass(frozen=True)
class KalmanForecastResult:
    """Auditable multi-step state-space forecast result."""

    forecast: np.ndarray
    forecast_cov: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    index: pd.Index
    filter_result: KalmanFilterResult
    alpha: float
    horizon: int

    def __post_init__(self) -> None:
        arrays = {
            "forecast": np.asarray(self.forecast, dtype=float).copy(),
            "forecast_cov": np.asarray(self.forecast_cov, dtype=float).copy(),
            "lower": np.asarray(self.lower, dtype=float).copy(),
            "upper": np.asarray(self.upper, dtype=float).copy(),
        }
        for name, value in arrays.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "index", pd.Index(self.index).copy())
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "horizon", int(self.horizon))
        if not isinstance(self.filter_result, KalmanFilterResult):
            raise TypeError("filter_result must be a KalmanFilterResult")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if len(self.index) != self.horizon:
            raise ValueError("index length must equal horizon")
        if self.forecast.ndim != 2:
            raise ValueError("forecast must be two-dimensional")
        if self.forecast.shape[0] != self.horizon:
            raise ValueError("forecast length must equal horizon")
        n_obs = self.forecast.shape[1]
        if self.forecast_cov.shape != (self.horizon, n_obs, n_obs):
            raise ValueError("forecast_cov has incompatible shape")
        if self.lower.shape != self.forecast.shape or self.upper.shape != self.forecast.shape:
            raise ValueError("prediction interval arrays have incompatible shape")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1")

    @property
    def standard_error(self) -> np.ndarray:
        """Marginal standard errors for each forecast horizon and observation."""
        return np.sqrt(np.maximum(np.diagonal(self.forecast_cov, axis1=1, axis2=2), 0.0))

    @property
    def prediction_interval(self) -> tuple[np.ndarray, np.ndarray]:
        """Return lower and upper marginal prediction limits."""
        return self.lower, self.upper


def _as_observations(observations: np.ndarray | Iterable[float]):
    if isinstance(observations, pd.Series):
        values = observations.to_numpy(dtype=float, copy=True).reshape(-1, 1)
        source_index: object = observations.index.copy()
    elif isinstance(observations, pd.DataFrame):
        values = observations.to_numpy(dtype=float, copy=True)
        source_index = observations.index.copy()
    else:
        values = np.asarray(observations, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        source_index = None
    return values, source_index


def _future_index(source: object, horizon: int) -> pd.Index:
    if source is None:
        return pd.RangeIndex(start=0, stop=horizon)
    idx = pd.Index(source)
    if isinstance(idx, pd.DatetimeIndex) and len(idx):
        freq = idx.freq or idx.inferred_freq
        if freq is not None:
            offset = pd.tseries.frequencies.to_offset(freq)
            return pd.date_range(start=idx[-1] + offset, periods=horizon, freq=offset)
    if isinstance(idx, pd.RangeIndex) and len(idx):
        step = idx.step
        start = idx[-1] + step
        return pd.RangeIndex(start=start, stop=start + step * horizon, step=step)
    if len(idx) and np.issubdtype(idx.dtype, np.integer):
        step = int(idx[-1] - idx[-2]) if len(idx) > 1 else 1
        return pd.Index([int(idx[-1]) + step * (i + 1) for i in range(horizon)])
    return pd.RangeIndex(start=0, stop=horizon)


def kalman_forecast(
    observations: np.ndarray | Iterable[float],
    model: LinearStateSpace,
    *,
    steps: int,
    alpha: float = 0.05,
    filter_result: KalmanFilterResult | None = None,
) -> KalmanForecastResult:
    """Produce deterministic multi-step forecasts and marginal prediction intervals.

    Forecasts start from the final filtered state/covariance. State uncertainty
    is propagated with the transition equation and process covariance, then
    transformed to observation space with the design matrix and observation
    covariance. Historical missing values retain Stage 10 semantics.
    """
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")

    values, source_index = _as_observations(observations)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != model.n_obs:
        raise ValueError(f"observations must have shape (n, {model.n_obs})")
    if np.isinf(values).any():
        raise ValueError("observations must not contain infinite values")

    if filter_result is None:
        filter_result = kalman_filter(values, model)
    else:
        if filter_result.nobs != values.shape[0]:
            raise ValueError("filter_result is incompatible with observations")
        expected_mask = np.isfinite(values).astype(int)
        if not np.array_equal(expected_mask, filter_result.observed_dimensions):
            raise ValueError("filter_result is incompatible with observation missingness")
        if filter_result.filtered_state.shape[1] != model.n_state:
            raise ValueError("filter_result is incompatible with model state dimension")

    state = filter_result.filtered_state[-1].copy()
    covariance = filter_result.filtered_cov[-1].copy()
    forecast = np.empty((steps, model.n_obs), dtype=float)
    forecast_cov = np.empty((steps, model.n_obs, model.n_obs), dtype=float)

    F = model.transition
    H = model.design
    Q = model.state_cov
    R = model.observation_cov

    for h in range(steps):
        state = F @ state + model.state_intercept
        propagated = F @ covariance @ F.T + Q
        covariance = 0.5 * (propagated + propagated.T)
        mean = H @ state + model.observation_intercept
        obs_cov = H @ covariance @ H.T + R
        forecast[h] = mean
        forecast_cov[h] = 0.5 * (obs_cov + obs_cov.T)

    z = float(norm.ppf(1.0 - alpha / 2.0))
    se = np.sqrt(np.maximum(np.diagonal(forecast_cov, axis1=1, axis2=2), 0.0))
    lower = forecast - z * se
    upper = forecast + z * se

    return KalmanForecastResult(
        forecast=forecast,
        forecast_cov=forecast_cov,
        lower=lower,
        upper=upper,
        index=_future_index(source_index, steps),
        filter_result=filter_result,
        alpha=alpha,
        horizon=steps,
    )
