"""Filtering, smoothing, decomposition and deterministic seasonality tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose, STL

from .series import TimeSeries


def _series(y: TimeSeries | Iterable[float]) -> pd.Series:
    if isinstance(y, TimeSeries):
        return pd.Series(y.values, index=y.index, name=y.name, dtype=float)
    return pd.Series(np.asarray(list(y), dtype=float), dtype=float)


def moving_average(y: TimeSeries | Iterable[float], window: int, *, centered: bool = False) -> TimeSeries:
    """Apply a moving-average filter/smoother."""
    if window < 1:
        raise ValueError("window must be positive")
    s = _series(y)
    values = s.rolling(window=window, center=centered, min_periods=window).mean().to_numpy()
    return TimeSeries(values, index=tuple(s.index), name=f"MA({window})", frequency=getattr(y, "frequency", None))


def weighted_moving_average(y: TimeSeries | Iterable[float], weights: Iterable[float]) -> TimeSeries:
    """Apply a finite weighted moving-average filter."""
    w = np.asarray(list(weights), dtype=float)
    if w.ndim != 1 or w.size < 1 or np.isclose(w.sum(), 0):
        raise ValueError("weights must be one-dimensional and sum to a non-zero value")
    w = w / w.sum()
    s = _series(y)
    values = np.full(s.size, np.nan)
    for i in range(w.size - 1, s.size):
        values[i] = float(np.dot(s.iloc[i - w.size + 1 : i + 1], w))
    return TimeSeries(values, index=tuple(s.index), name="WMA", frequency=getattr(y, "frequency", None))


def exponential_smoothing(y: TimeSeries | Iterable[float], *, alpha: float = 0.2, initial: float | None = None) -> TimeSeries:
    """Apply simple exponential smoothing."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    s = _series(y).astype(float)
    if s.isna().any():
        raise ValueError("exponential_smoothing requires a complete series")
    out = np.empty(s.size)
    out[0] = float(s.iloc[0] if initial is None else initial)
    for t in range(1, s.size):
        out[t] = alpha * s.iloc[t] + (1 - alpha) * out[t - 1]
    return TimeSeries(out, index=tuple(s.index), name=f"SES({alpha:g})", frequency=getattr(y, "frequency", None))


def holt(y: TimeSeries | Iterable[float], *, damped: bool = False, alpha: float = 0.3, beta: float = 0.1, phi: float = 0.98) -> TimeSeries:
    """Compute Holt level-trend exponential smoothing."""
    s = _series(y)
    if s.isna().any() or s.size < 2:
        raise ValueError("Holt smoothing requires at least two complete observations")
    if not (0 < alpha <= 1 and 0 < beta <= 1 and (0 < phi <= 1)):
        raise ValueError("alpha, beta and phi must be in their standard ranges")
    level = float(s.iloc[0])
    trend = float(s.iloc[1] - s.iloc[0])
    fitted = np.empty(s.size)
    fitted[0] = level
    for t in range(1, s.size):
        fitted[t] = level + trend if not damped else level + phi * trend
        old_level = level
        level = alpha * s.iloc[t] + (1 - alpha) * (level + (phi * trend if damped else trend))
        trend = beta * (level - old_level) + (1 - beta) * (phi * trend if damped else trend)
    return TimeSeries(fitted, index=tuple(s.index), name="Holt", frequency=getattr(y, "frequency", None))


def holt_winters(y: TimeSeries | Iterable[float], *, period: int, trend: str = "add", seasonal: str = "add", seasonal_periods: int | None = None) -> pd.DataFrame:
    """Fit additive or multiplicative Holt-Winters exponential smoothing."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    s = _series(y)
    period = seasonal_periods or period
    model = ExponentialSmoothing(s, trend=trend, seasonal=seasonal, seasonal_periods=period)
    result = model.fit(optimized=True)
    frame = pd.DataFrame({"Actual": s, "Fitted": result.fittedvalues}, index=s.index)
    return frame


@dataclass(frozen=True)
class DecompositionResult:
    """Trend-seasonal-residual decomposition output."""

    observed: pd.Series
    trend: pd.Series
    seasonal: pd.Series
    resid: pd.Series
    mode: str

    @property
    def adjusted(self) -> pd.Series:
        """Return the seasonally adjusted series."""
        if self.mode == "additive":
            return self.observed - self.seasonal
        return self.observed / self.seasonal


def decompose(y: TimeSeries | Iterable[float], *, period: int, model: str = "additive", method: str = "classical") -> DecompositionResult:
    """Perform classical additive/multiplicative or STL decomposition."""
    if period < 2:
        raise ValueError("period must be at least 2")
    s = _series(y)
    if method == "stl":
        result = STL(s, period=period, robust=True).fit()
        return DecompositionResult(s, result.trend, result.seasonal, result.resid, "additive")
    result = seasonal_decompose(s, model=model, period=period, extrapolate_trend=False)
    return DecompositionResult(s, result.trend, result.seasonal, result.resid, model)


def seasonal_difference(y: TimeSeries, period: int) -> TimeSeries:
    """Apply the seasonal difference operator Δ_s=(1-L^s)."""
    if period < 1 or period >= y.nobs:
        raise ValueError("period must be positive and smaller than the series length")
    values = y.values[period:] - y.values[:-period]
    index = y.index[period:] if y.index is not None else None
    return TimeSeries(values, index=index, name=f"D_s({y.name})", frequency=y.frequency)


def seasonal_indices(y: TimeSeries | Iterable[float], period: int, *, additive: bool = True) -> pd.Series:
    """Estimate average seasonal indices by within-period means."""
    s = _series(y)
    if s.size < 2 * period:
        raise ValueError("at least two full seasons are required")
    groups = [s.iloc[i::period] for i in range(period)]
    if additive:
        overall = float(s.mean())
        values = np.array([float(g.mean() - overall) for g in groups])
        values -= values.mean()
    else:
        overall = float(s.mean())
        values = np.array([float(g.mean() / overall) for g in groups])
        values /= values.mean()
    return pd.Series(values, index=np.arange(1, period + 1), name="Seasonal Index")


def fisher_seasonality_test(y: TimeSeries | Iterable[float], period: int, *, alpha: float = 0.05) -> dict[str, float | bool | str]:
    """Run a one-way ANOVA/Fisher test of seasonal means."""
    s = _series(y).dropna()
    groups = [s.iloc[i::period].to_numpy() for i in range(period)]
    groups = [g for g in groups if g.size >= 2]
    if len(groups) < 2:
        raise ValueError("not enough seasonal groups")
    statistic, pvalue = stats.f_oneway(*groups)
    return {"test": "Fisher seasonal ANOVA", "F-statistic": float(statistic), "p-value": float(pvalue), "reject_seasonality_null": bool(pvalue < alpha), "conclusion": "Seasonal effects are significant" if pvalue < alpha else "No statistically significant seasonal effect detected"}
