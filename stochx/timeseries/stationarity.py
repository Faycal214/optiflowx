"""Stationarity, unit-root, TS/DS and differencing diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from .series import TimeSeries


@dataclass(frozen=True)
class UnitRootResult:
    """Structured unit-root result with course-oriented interpretation."""

    test: str
    statistic: float
    pvalue: float | None
    critical_values: dict[str, float]
    regression: str
    lags: int
    nobs: int
    null_hypothesis: str
    conclusion: str

    def summary(self) -> str:
        """Render an EViews-like unit-root test table."""
        lines = [
            f"{self.test}",
            "=" * 72,
            f"Test equation: {self.regression}",
            f"Included observations: {self.nobs}",
            f"Lagged differences: {self.lags}",
            f"Test statistic: {self.statistic:.6f}",
        ]
        if self.pvalue is not None and np.isfinite(self.pvalue):
            lines.append(f"Prob.*: {self.pvalue:.6f}")
        lines.append("")
        lines.append("Critical Values:")
        for level, value in self.critical_values.items():
            lines.append(f"{level}: {value:.6f}")
        lines.append("")
        lines.append(f"Null: {self.null_hypothesis}")
        lines.append(f"Conclusion: {self.conclusion}")
        return "\n".join(lines)


def _values(y: TimeSeries | Iterable[float]) -> np.ndarray:
    x = np.asarray(y.values if isinstance(y, TimeSeries) else list(y), dtype=float).reshape(-1)
    x = x[~np.isnan(x)]
    if x.size < 10:
        raise ValueError("at least 10 observations are recommended for unit-root testing")
    if np.any(np.isinf(x)):
        raise ValueError("series must contain no infinite values")
    return x


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
    constant), ``c`` (constant), and ``ct`` (constant plus trend).  Lagged
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
    if statistic < critical["5%"]:
        conclusion = "Reject the unit-root null at 5%; the series is consistent with stationarity under the selected deterministic specification."
    else:
        conclusion = "Do not reject the unit-root null at 5%; the series is consistent with non-stationarity under the selected deterministic specification."
    return UnitRootResult("Augmented Dickey-Fuller Test", float(statistic), float(pvalue), {str(k): float(v) for k, v in critical.items()}, regression, int(usedlag), int(nobs), "The series contains a unit root (non-stationary).", conclusion)


def dickey_fuller_sequential(
    y: TimeSeries | Iterable[float],
    *,
    max_lags: int | None = None,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Apply the course's sequential DF/ADF strategy over no-constant, constant, and trend specifications."""
    results = {}
    for regression, label in [("ct", "model_3_trend_intercept"), ("c", "model_2_intercept"), ("n", "model_1_none")]:
        results[label] = adf(y, regression=regression, lags=max_lags, autolag="AIC", alpha=alpha)
    ordered = [results["model_3_trend_intercept"], results["model_2_intercept"], results["model_1_none"]]
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
    """Run the KPSS stationarity test as a complementary diagnostic."""
    x = _values(y)
    statistic, pvalue, lags, critical = kpss(x, regression=regression, nlags=nlags)
    conclusion = "Reject stationarity at 5%; evidence favors non-stationarity." if statistic > critical["5%"] else "Do not reject stationarity at 5%."
    return UnitRootResult("KPSS Test", float(statistic), float(pvalue), {str(k): float(v) for k, v in critical.items()}, regression, int(lags), int(x.size - lags), "The series is stationary.", conclusion)


def phillips_perron(
    y: TimeSeries | Iterable[float],
    *,
    trend: str = "c",
    lags: int | None = None,
) -> UnitRootResult:
    """Run the Phillips-Perron unit-root test when the optional ``arch`` backend is installed."""
    x = _values(y)
    if trend not in {"n", "c", "ct"}:
        raise ValueError("trend must be 'n', 'c', or 'ct'")
    try:
        from arch.unitroot import PhillipsPerron
    except ImportError as exc:
        raise ImportError("Phillips-Perron requires the optional 'arch' dependency. Install with: pip install arch") from exc
    test = PhillipsPerron(x, trend=trend, lags=lags)
    critical = {str(k): float(v) for k, v in test.critical_values.items()}
    conclusion = "Reject the unit-root null at 5%; evidence favors stationarity." if test.stat < critical["5%"] else "Do not reject the unit-root null at 5%; evidence favors non-stationarity."
    return UnitRootResult("Phillips-Perron Test", float(test.stat), float(test.pvalue), critical, trend, int(test.lags), int(test.nobs), "The series contains a unit root.", conclusion)


def difference(y: TimeSeries, order: int = 1, seasonal_period: int | None = None) -> TimeSeries:
    """Apply ordinary and/or seasonal differencing operators."""
    if order < 0:
        raise ValueError("order must be non-negative")
    result = y
    for _ in range(order):
        result = result.diff(1)
    if seasonal_period is not None:
        if seasonal_period < 1 or seasonal_period >= result.nobs:
            raise ValueError("seasonal_period must be positive and smaller than the series length")
        values = result.values[seasonal_period:] - result.values[:-seasonal_period]
        index = result.index[seasonal_period:] if result.index is not None else None
        result = TimeSeries(values, index=index, name=f"DS{seasonal_period}({result.name})", frequency=result.frequency)
    return result


def classify_ts_ds(y: TimeSeries | Iterable[float]) -> dict[str, object]:
    """Classify a series as TS-like, DS-like, or inconclusive using the sequential test workflow."""
    report = dickey_fuller_sequential(y)
    nature = str(report["nature"])
    return {**report, "is_ts_candidate": "trend" in nature or "intercept" in nature, "is_ds_candidate": "DS" in nature}


def trend_regression(y: TimeSeries | Iterable[float], *, degree: int = 1) -> pd.DataFrame:
    """Estimate a deterministic polynomial trend and return coefficients and residuals."""
    x = _values(y)
    t = np.arange(1, x.size + 1, dtype=float)
    X = np.column_stack([t**j for j in range(degree + 1)])
    beta = np.linalg.lstsq(X, x, rcond=None)[0]
    fitted = X @ beta
    residual = x - fitted
    rows = [{"Term": "Intercept" if j == 0 else f"Trend^{j}", "Coefficient": float(beta[j])} for j in range(degree + 1)]
    return pd.DataFrame(rows).assign(R2=float(1 - np.sum(residual**2) / np.sum((x - x.mean())**2)))
