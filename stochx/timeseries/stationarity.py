"""Stationarity and unit-root diagnostics for time-series analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, phillips_perron

from .series import TimeSeries


@dataclass(frozen=True)
class UnitRootResult:
    """Structured output for a unit-root or stationarity test."""

    test: str
    statistic: float
    pvalue: float
    critical_values: dict[str, float]
    regression: str
    lags: int
    nobs: int
    null_hypothesis: str
    conclusion: str

    def summary(self) -> str:
        """Return an EViews-style textual test summary."""
        lines = [
            self.test,
            "=" * len(self.test),
            f"Regression: {self.regression}",
            f"Included lags: {self.lags}",
            f"Included observations: {self.nobs}",
            f"Test Statistic: {self.statistic:.6f}",
            f"Prob.*: {self.pvalue:.6f}",
            "Critical Values:",
        ]
        lines.extend(f"  {key}: {value:.6f}" for key, value in self.critical_values.items())
        lines.append("")
        lines.append(f"Null Hypothesis: {self.null_hypothesis}")
        lines.append(f"Conclusion: {self.conclusion}")
        return "\n".join(lines)

    def interpret(self) -> str:
        """Return the course-oriented interpretation."""
        return self.conclusion


def _values(y: TimeSeries | Iterable[float]) -> np.ndarray:
    values = np.asarray(y.values if isinstance(y, TimeSeries) else list(y), dtype=float)
    if values.ndim != 1:
        raise ValueError("series must be one-dimensional")
    values = values[~np.isnan(values)]
    if values.size < 10:
        raise ValueError("at least 10 non-missing observations are required")
    if np.any(np.isinf(values)):
        raise ValueError("series must not contain infinite observations")
    return values


def difference(y: TimeSeries | Iterable[float], order: int = 1) -> TimeSeries:
    """Return an ordinary difference of the requested order."""
    if not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer")
    if isinstance(y, TimeSeries):
        return y.diff(order)
    values = _values(y)
    for _ in range(order):
        values = np.diff(values)
    return TimeSeries(values, name=f"D({getattr(y, 'name', 'series')},{order})")


def adf(
    y: TimeSeries | Iterable[float],
    *,
    regression: str = "ct",
    lags: int | None = None,
    autolag: str | None = "AIC",
    alpha: float = 0.05,
) -> UnitRootResult:
    """Run the Augmented Dickey-Fuller test with configurable deterministic terms.

    ``regression`` follows the three course specifications: ``n`` (no
    constant), ``c`` (constant), and ``ct`` (constant plus trend). Lagged
    differences can be selected explicitly or by an information criterion,
    following the course strategy for whitening the innovations.
    """
    x = _values(y)
    if regression not in {"n", "c", "ct"}:
        raise ValueError("regression must be 'n', 'c', or 'ct'")
    if lags is not None and (not isinstance(lags, int) or lags < 0):
        raise ValueError("lags must be a non-negative integer or None")
    result = adfuller(x, regression=regression, maxlag=lags, autolag=autolag)
    if len(result) == 5:
        statistic, pvalue, usedlag, nobs, critical = result
    elif len(result) == 6:
        statistic, pvalue, usedlag, nobs, critical, _ = result
    else:
        raise RuntimeError(f"unexpected statsmodels ADF result length: {len(result)}")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    critical_values = {str(k): float(v) for k, v in critical.items()}
    if statistic < critical_values["5%"]:
        conclusion = "Reject the unit-root null at 5%; the series is consistent with stationarity under the selected deterministic specification."
    else:
        conclusion = "Do not reject the unit-root null at 5%; the series is consistent with non-stationarity under the selected deterministic specification."
    return UnitRootResult(
        "Augmented Dickey-Fuller Test",
        float(statistic),
        float(pvalue),
        critical_values,
        regression,
        int(usedlag),
        int(nobs),
        "The series contains a unit root (non-stationary).",
        conclusion,
    )


def dickey_fuller_sequential(
    y: TimeSeries | Iterable[float],
    *,
    max_lags: int | None = None,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Apply the course's sequential DF/ADF strategy over deterministic specifications."""
    results: dict[str, UnitRootResult] = {}
    for regression, label in [
        ("ct", "model_3_trend_intercept"),
        ("c", "model_2_intercept"),
        ("n", "model_1_none"),
    ]:
        results[label] = adf(y, regression=regression, lags=max_lags, autolag="AIC", alpha=alpha)

    ordered = [
        results["model_3_trend_intercept"],
        results["model_2_intercept"],
        results["model_1_none"],
    ]
    selected = ordered[-1]
    if ordered[0].statistic < ordered[0].critical_values["5%"]:
        selected = ordered[0]
        nature = "stationary with deterministic trend/intercept component"
    elif ordered[1].statistic < ordered[1].critical_values["5%"]:
        selected = ordered[1]
        nature = "stationary around an intercept"
    elif ordered[2].statistic < ordered[2].critical_values["5%"]:
        selected = ordered[2]
        nature = "stationary around zero"
    else:
        nature = "difference-stationary / integrated candidate (DS)"
    return {"tests": results, "selected": selected, "nature": nature}


def kpss_test(
    y: TimeSeries | Iterable[float],
    *,
    regression: str = "c",
    nlags: str | int = "auto",
    alpha: float = 0.05,
) -> UnitRootResult:
    """Run the KPSS stationarity test."""
    x = _values(y)
    statistic, pvalue, lags, critical = kpss(x, regression=regression, nlags=nlags)
    critical_values = {str(k): float(v) for k, v in critical.items()}
    if pvalue < alpha:
        conclusion = "Reject the stationarity null at the selected level; evidence indicates non-stationarity."
    else:
        conclusion = "Do not reject the stationarity null at the selected level."
    return UnitRootResult(
        "KPSS Test",
        float(statistic),
        float(pvalue),
        critical_values,
        regression,
        int(lags),
        int(x.size),
        "The series is stationary under the selected deterministic specification.",
        conclusion,
    )


def phillips_perron(
    y: TimeSeries | Iterable[float],
    *,
    trend: str = "c",
    lags: int | None = None,
    alpha: float = 0.05,
) -> UnitRootResult:
    """Run the Phillips-Perron unit-root test when available."""
    x = _values(y)
    result = phillips_perron(x, trend=trend, lags=lags)
    statistic, pvalue, critical = result[0], result[1], result[2]
    critical_values = {str(k): float(v) for k, v in critical.items()}
    conclusion = (
        "Reject the unit-root null at the selected level."
        if statistic < critical_values["5%"]
        else "Do not reject the unit-root null at the selected level."
    )
    used_lags = int(result[3]) if len(result) > 3 else (0 if lags is None else lags)
    return UnitRootResult(
        "Phillips-Perron Test",
        float(statistic),
        float(pvalue),
        critical_values,
        trend,
        used_lags,
        int(x.size),
        "The series contains a unit root (non-stationary).",
        conclusion,
    )


def trend_regression(y: TimeSeries | Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return fitted values and residuals from a deterministic linear trend."""
    x = _values(y)
    t = np.arange(1, x.size + 1, dtype=float)
    design = np.column_stack([np.ones_like(t), t])
    beta = np.linalg.lstsq(design, x, rcond=None)[0]
    fitted = design @ beta
    return fitted, x - fitted


def classify_ts_ds(y: TimeSeries | Iterable[float]) -> str:
    """Classify a series using the course-oriented sequential DF strategy."""
    result = dickey_fuller_sequential(y)
    return str(result["nature"])
