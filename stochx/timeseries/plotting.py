"""Plotting helpers for the StochX time-series workflow."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt


def _values(y):
    return np.asarray(y.values if hasattr(y, "values") else list(y), float)


def plot_series(y, *, ax=None, title: str | None = None, show: bool = True):
    """Plot a time series against its supplied index or integer time."""
    x = _values(y)
    idx = getattr(y, "index", None)
    if idx is None:
        idx = np.arange(x.size)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    ax.plot(idx, x)
    ax.set_title(title or getattr(y, "name", "Series"))
    ax.set_xlabel("Time")
    ax.set_ylabel(getattr(y, "name", "Y"))
    ax.grid(True, alpha=0.25)
    if show:
        plt.show()
    return ax


def plot_correlogram(result, *, ax=None, title: str | None = None, show: bool = True):
    """Plot ACF or PACF bars with confidence bands."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0.0, linewidth=0.8)
    ax.vlines(result.lags, 0.0, result.values, linewidth=1.2)
    ax.plot(result.lags, result.upper, linestyle="--", linewidth=0.8)
    ax.plot(result.lags, result.lower, linestyle="--", linewidth=0.8)
    ax.set_title(title or ("ACF" if result.__class__.__name__.startswith("ACF") else "PACF"))
    ax.set_xlabel("Lag")
    if show:
        plt.show()
    return ax


def plot_forecast(history, forecast, *, ax=None, title: str = "Forecast", show: bool = True):
    """Plot historical observations and point/interval forecasts."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    h = _values(history)
    idx = np.arange(h.size)
    ax.plot(idx, h, label="Actual")
    if hasattr(forecast, "index"):
        fidx = np.arange(h.size, h.size + len(forecast))
        mean = np.asarray(forecast["Forecast"], float)
        ax.plot(fidx, mean, label="Forecast")
        if "Lower" in forecast and "Upper" in forecast:
            ax.fill_between(fidx, forecast["Lower"], forecast["Upper"], alpha=0.2, label="Prediction interval")
    else:
        mean = np.asarray(forecast, float)
        fidx = np.arange(h.size, h.size + mean.size)
        ax.plot(fidx, mean, label="Forecast")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.legend()
    if show:
        plt.show()
    return ax


def plot_decomposition(result, *, show: bool = True):
    """Plot observed, trend, seasonal and residual components."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    result.observed.plot(ax=axes[0], title="Observed")
    result.trend.plot(ax=axes[1], title="Trend-Cycle")
    result.seasonal.plot(ax=axes[2], title="Seasonal")
    result.resid.plot(ax=axes[3], title="Residual")
    fig.tight_layout()
    if show:
        plt.show()
    return axes
