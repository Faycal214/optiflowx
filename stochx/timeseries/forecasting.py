"""Forecast evaluation and prediction utilities for course TPs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class ForecastMetrics:
    """Standard forecast error criteria used in the course."""

    mse: float
    rmse: float
    mae: float
    mape: float
    me: float

    def as_dict(self) -> dict[str, float]:
        """Return metrics as a dictionary."""
        return {"MSE": self.mse, "RMSE": self.rmse, "MAE": self.mae, "MAPE": self.mape, "ME": self.me}


def metrics(actual: Iterable[float], forecast: Iterable[float], *, epsilon: float = 1e-12) -> ForecastMetrics:
    """Compute MSE, RMSE, MAE, MAPE and mean error."""
    y = np.asarray(list(actual), dtype=float)
    f = np.asarray(list(forecast), dtype=float)
    if y.shape != f.shape:
        raise ValueError("actual and forecast must have the same shape")
    mask = np.isfinite(y) & np.isfinite(f)
    y, f = y[mask], f[mask]
    if y.size == 0:
        raise ValueError("no finite paired observations")
    e = y - f
    nonzero = np.abs(y) > epsilon
    mape = float(np.mean(np.abs(e[nonzero] / y[nonzero])) * 100) if np.any(nonzero) else np.nan
    mse = float(np.mean(e**2))
    return ForecastMetrics(mse, float(np.sqrt(mse)), float(np.mean(np.abs(e))), mape, float(np.mean(e)))


def prediction_interval(mean: Iterable[float], variance: Iterable[float] | float, *, alpha: float = 0.05, distribution: str = "normal") -> pd.DataFrame:
    """Construct point and interval forecasts from predictive variance."""
    mu = np.asarray(list(mean), dtype=float)
    var = np.asarray(variance if np.ndim(variance) else np.full(mu.size, variance), dtype=float)
    if var.shape != mu.shape or np.any(var < 0):
        raise ValueError("variance must match mean shape and be non-negative")
    if distribution == "normal":
        critical = stats.norm.ppf(1 - alpha / 2)
    else:
        raise ValueError("distribution must currently be 'normal'")
    se = np.sqrt(var)
    return pd.DataFrame({"Forecast": mu, "Std. Error": se, "Lower": mu - critical * se, "Upper": mu + critical * se})


def restore_differences(last_levels: Iterable[float], differenced_forecast: Iterable[float], *, order: int = 1, seasonal_period: int | None = None) -> np.ndarray:
    """Reconstruct level forecasts after ordinary differencing."""
    history = list(map(float, last_levels))
    forecasts = np.asarray(list(differenced_forecast), dtype=float)
    if order < 0:
        raise ValueError("order must be non-negative")
    if seasonal_period is not None:
        raise NotImplementedError("combined inverse ordinary and seasonal differencing is reserved for the SARIMA reconstruction layer")
    if order == 0:
        return forecasts.copy()
    for _ in range(order):
        current = history[-1] if history else 0.0
        rebuilt = np.empty(forecasts.size)
        for i, value in enumerate(forecasts):
            current = current + value
            rebuilt[i] = current
        forecasts = rebuilt
        history = list(rebuilt)
    return forecasts


def naive_forecast(history: Iterable[float], steps: int = 1) -> np.ndarray:
    """Produce a random-walk/naive forecast."""
    x = np.asarray(list(history), float)
    if x.size == 0 or steps < 1:
        raise ValueError("history must be non-empty and steps positive")
    return np.full(steps, x[-1], dtype=float)


def drift_forecast(history: Iterable[float], steps: int = 1) -> np.ndarray:
    """Produce a random-walk-with-drift forecast."""
    x = np.asarray(list(history), float)
    if x.size < 2 or steps < 1:
        raise ValueError("at least two history observations are required")
    drift = (x[-1] - x[0]) / (x.size - 1)
    return x[-1] + drift * np.arange(1, steps + 1)
