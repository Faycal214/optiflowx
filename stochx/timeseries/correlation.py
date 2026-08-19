"""EViews-style autocorrelation and partial-autocorrelation calculations.

Stage 8.2 freezes the numerical core used by the correlogram and later
Box-Jenkins identification workflow:

* AC uses EViews' common-overall-mean convention.
* PAC uses the recursive Box-Jenkins / Durbin-Levinson construction.

Stage 8.4 freezes a common complete-observation policy: NaNs are removed
once before AC/PAC calculation, while infinite observations are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .series import TimeSeries


@dataclass(frozen=True)
class ACFResult:
    """Autocorrelation function and large-sample confidence bands."""

    lags: np.ndarray
    values: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    nobs: int
    series_name: str
    missing_count: int = 0

    def significant(self) -> np.ndarray:
        return (self.values < self.lower) | (self.values > self.upper)

    def table(self) -> list[dict[str, float | int | bool]]:
        significant = self.significant()
        return [
            {"Lag": int(lag), "AC": float(value), "Lower": float(lower),
             "Upper": float(upper), "Significant": bool(sig)}
            for lag, value, lower, upper, sig in zip(
                self.lags, self.values, self.lower, self.upper, significant
            )
        ]

    def summary(self) -> str:
        lines = [
            f"Autocorrelation for {self.series_name}",
            f"Included observations: {self.nobs}",
            f"Excluded missing observations: {self.missing_count}",
            "Lag       AC        Lower      Upper",
            "---------------------------------------",
        ]
        for row in self.table():
            marker = " *" if row["Significant"] else ""
            lines.append(
                f"{row['Lag']:>3d}   {row['AC']:>9.4f}   "
                f"{row['Lower']:>9.4f}   {row['Upper']:>9.4f}{marker}"
            )
        return "\n".join(lines)

    def interpret(self) -> str:
        significant = self.significant()
        sig_lags = [int(lag) for lag in self.lags[1:] if significant[lag]]
        if not sig_lags:
            return "No non-zero autocorrelation is outside the 5% confidence bands."
        return (
            "Significant autocorrelations occur at lags "
            + ", ".join(map(str, sig_lags))
            + ". Inspect the ACF together with the PACF before selecting an ARMA order."
        )


@dataclass(frozen=True)
class PACFResult:
    """Partial autocorrelation function and large-sample confidence bands."""

    lags: np.ndarray
    values: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    nobs: int
    series_name: str
    missing_count: int = 0

    def significant(self) -> np.ndarray:
        return (self.values < self.lower) | (self.values > self.upper)

    def table(self) -> list[dict[str, float | int | bool]]:
        significant = self.significant()
        return [
            {"Lag": int(lag), "PAC": float(value), "Lower": float(lower),
             "Upper": float(upper), "Significant": bool(sig)}
            for lag, value, lower, upper, sig in zip(
                self.lags, self.values, self.lower, self.upper, significant
            )
        ]

    def summary(self) -> str:
        lines = [
            f"Partial Autocorrelation for {self.series_name}",
            f"Included observations: {self.nobs}",
            f"Excluded missing observations: {self.missing_count}",
            "Lag       PAC       Lower      Upper",
            "---------------------------------------",
        ]
        for row in self.table():
            marker = " *" if row["Significant"] else ""
            lines.append(
                f"{row['Lag']:>3d}   {row['PAC']:>9.4f}   "
                f"{row['Lower']:>9.4f}   {row['Upper']:>9.4f}{marker}"
            )
        return "\n".join(lines)

    def interpret(self) -> str:
        significant = self.significant()
        sig_lags = [int(lag) for lag in self.lags[1:] if significant[lag]]
        if not sig_lags:
            return "No non-zero partial autocorrelation is outside the 5% confidence bands."
        max_lag = max(sig_lags)
        return (
            "Significant partial autocorrelations occur at lags "
            + ", ".join(map(str, sig_lags))
            + f". The largest significant lag is {max_lag}; use this as an AR-order clue, "
            "then validate the candidate model with residual diagnostics."
        )


def _clean_values(series: TimeSeries | np.ndarray | list[float]) -> tuple[np.ndarray, str, int]:
    """Apply Stage 8.4 common complete-observation preprocessing once."""
    if hasattr(series, "values"):
        values = np.asarray(series.values, dtype=float)
        name = getattr(series, "name", "series")
    else:
        values = np.asarray(series, dtype=float)
        name = "series"
    if values.ndim != 1:
        raise ValueError("series must be one-dimensional")
    if np.any(np.isinf(values)):
        raise ValueError("series must not contain infinite observations")
    missing_count = int(np.isnan(values).sum())
    values = values[~np.isnan(values)]
    if values.size < 2:
        raise ValueError("at least two non-missing observations are required")
    return values, str(name), missing_count


def _validate_inputs(nobs: int, nlags: int | None, alpha: float) -> int:
    if nlags is None:
        nlags = min(36, nobs // 4)
    if not isinstance(nlags, int) or isinstance(nlags, bool) or nlags < 0:
        raise ValueError("nlags must be a non-negative integer")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    return min(nlags, nobs - 1)


def acf(series: TimeSeries | np.ndarray | list[float], nlags: int | None = None, *, alpha: float = 0.05) -> ACFResult:
    """Estimate the EViews-style ACF after one common missing-value pass."""
    values, name, missing_count = _clean_values(series)
    nobs = values.size
    nlags = _validate_inputs(nobs, nlags, alpha)
    centered = values - values.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        raise ValueError("ACF is undefined for a constant series")

    result = np.empty(nlags + 1, dtype=float)
    result[0] = 1.0
    for lag in range(1, nlags + 1):
        result[lag] = float(np.dot(centered[lag:], centered[:-lag]) / denominator)

    z = _normal_critical_value(alpha)
    bound = z / np.sqrt(nobs)
    lags = np.arange(nlags + 1, dtype=int)
    lower = np.full(nlags + 1, -bound, dtype=float)
    upper = np.full(nlags + 1, bound, dtype=float)
    return ACFResult(lags, result, lower, upper, nobs, name, missing_count)


def _recursive_pacf(ac_values: np.ndarray) -> np.ndarray:
    """Compute PACF recursively from AC using the Box-Jenkins recursion."""
    nlags = len(ac_values) - 1
    pac_values = np.ones(nlags + 1, dtype=float)
    if nlags == 0:
        return pac_values
    previous = np.empty(0, dtype=float)
    for k in range(1, nlags + 1):
        numerator = float(ac_values[k])
        if k > 1:
            numerator -= float(np.dot(previous, ac_values[k - 1 : 0 : -1]))
        denominator = 1.0
        if k > 1:
            denominator -= float(np.dot(previous, ac_values[1:k]))
        if abs(denominator) <= np.finfo(float).eps:
            raise ValueError(f"PACF recursion is singular at lag {k}")
        phi_kk = numerator / denominator
        current = np.empty(k, dtype=float)
        if k == 1:
            current[0] = phi_kk
        else:
            current[:-1] = previous - phi_kk * previous[::-1]
            current[-1] = phi_kk
        previous = current
        pac_values[k] = phi_kk
    return pac_values


def pacf(series: TimeSeries | np.ndarray | list[float], nlags: int | None = None, *, alpha: float = 0.05) -> PACFResult:
    """Estimate the recursive EViews/course PACF after one common missing-value pass."""
    values, name, missing_count = _clean_values(series)
    nobs = values.size
    nlags = _validate_inputs(nobs, nlags, alpha)
    ac_values = acf(values, nlags=nlags, alpha=alpha).values
    pac_values = _recursive_pacf(ac_values)
    z = _normal_critical_value(alpha)
    bound = z / np.sqrt(nobs)
    lags = np.arange(nlags + 1, dtype=int)
    lower = np.full(nlags + 1, -bound, dtype=float)
    upper = np.full(nlags + 1, bound, dtype=float)
    return PACFResult(lags, pac_values, lower, upper, nobs, name, missing_count)


def _normal_critical_value(alpha: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(1.0 - alpha / 2.0))
