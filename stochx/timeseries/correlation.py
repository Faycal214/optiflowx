"""ACF and PACF calculations following the course identification workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .series import TimeSeries


@dataclass(frozen=True)
class ACFResult:
    """Autocorrelation function and its large-sample confidence bands."""

    lags: np.ndarray
    values: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    nobs: int
    series_name: str

    def significant(self) -> np.ndarray:
        """Return a boolean mask for autocorrelations outside the bands."""

        return (self.values < self.lower) | (self.values > self.upper)

    def table(self) -> list[dict[str, float | int | bool]]:
        """Return rows suitable for tabular display or export."""

        return [
            {
                "Lag": int(lag),
                "AC": float(value),
                "Lower": float(lower),
                "Upper": float(upper),
                "Significant": bool(significant),
            }
            for lag, value, lower, upper, significant in zip(
                self.lags,
                self.values,
                self.lower,
                self.upper,
                self.significant(),
            )
        ]

    def summary(self) -> str:
        """Return an EViews-like correlogram table."""

        lines = [
            f"Autocorrelation for {self.series_name}",
            f"Included observations: {self.nobs}",
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
        """Provide a first-pass identification interpretation."""

        sig_lags = [int(lag) for lag in self.lags[1:] if self.significant()[lag]]
        if not sig_lags:
            return "No non-zero autocorrelation is outside the 5% confidence bands."
        return (
            "Significant autocorrelations occur at lags "
            + ", ".join(map(str, sig_lags))
            + ". Inspect the ACF together with the PACF before selecting an ARMA order."
        )


@dataclass(frozen=True)
class PACFResult:
    """Partial autocorrelation function and its large-sample confidence bands."""

    lags: np.ndarray
    values: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    nobs: int
    series_name: str

    def significant(self) -> np.ndarray:
        """Return a boolean mask for partial autocorrelations outside the bands."""

        return (self.values < self.lower) | (self.values > self.upper)

    def table(self) -> list[dict[str, float | int | bool]]:
        """Return rows suitable for tabular display or export."""

        return [
            {
                "Lag": int(lag),
                "PAC": float(value),
                "Lower": float(lower),
                "Upper": float(upper),
                "Significant": bool(significant),
            }
            for lag, value, lower, upper, significant in zip(
                self.lags,
                self.values,
                self.lower,
                self.upper,
                self.significant(),
            )
        ]

    def summary(self) -> str:
        """Return an EViews-like partial correlogram table."""

        lines = [
            f"Partial Autocorrelation for {self.series_name}",
            f"Included observations: {self.nobs}",
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
        """Provide a first-pass AR-order identification interpretation."""

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


def _clean_values(series: TimeSeries | np.ndarray | list[float]) -> tuple[np.ndarray, str]:
    if hasattr(series, "values"):
        values = np.asarray(series.values, dtype=float)
        name = getattr(series, "name", "series")
    else:
        values = np.asarray(series, dtype=float)
        name = "series"
    if values.ndim != 1:
        raise ValueError("series must be one-dimensional")
    values = values[~np.isnan(values)]
    if values.size < 2:
        raise ValueError("at least two non-missing observations are required")
    if np.any(np.isinf(values)):
        raise ValueError("series must not contain infinite observations")
    return values, str(name)


def acf(
    series: TimeSeries | np.ndarray | list[float],
    nlags: int | None = None,
    *,
    alpha: float = 0.05,
) -> ACFResult:
    """Estimate the autocorrelation function for lags ``0..nlags``.

    The estimator uses the empirical autocovariance convention from the
    course material, with a lag-specific denominator ``T-k``. Confidence
    bands use the standard large-sample ``± z_(1-alpha/2)/sqrt(T)`` rule.
    """

    values, name = _clean_values(series)
    nobs = values.size
    if nlags is None:
        nlags = min(36, nobs // 4)
    if not isinstance(nlags, int) or nlags < 0:
        raise ValueError("nlags must be a non-negative integer")
    nlags = min(nlags, nobs - 1)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")

    centered = values - values.mean()
    variance = np.dot(centered, centered) / nobs
    if variance <= 0:
        raise ValueError("ACF is undefined for a constant series")

    result = np.empty(nlags + 1, dtype=float)
    result[0] = 1.0
    for lag in range(1, nlags + 1):
        covariance = np.dot(centered[lag:], centered[:-lag]) / (nobs - lag)
        result[lag] = covariance / variance

    z = _normal_critical_value(alpha)
    bound = z / np.sqrt(nobs)
    lags = np.arange(nlags + 1)
    lower = np.full(nlags + 1, -bound, dtype=float)
    upper = np.full(nlags + 1, bound, dtype=float)
    return ACFResult(lags, result, lower, upper, nobs, name)


def pacf(
    series: TimeSeries | np.ndarray | list[float],
    nlags: int | None = None,
    *,
    alpha: float = 0.05,
) -> PACFResult:
    """Estimate the PACF through the Yule-Walker equations.

    For each lag ``k``, the returned PACF is the final coefficient in the
    linear projection on the first ``k`` lags, matching the course definition.
    """

    values, name = _clean_values(series)
    nobs = values.size
    if nlags is None:
        nlags = min(36, nobs // 4)
    if not isinstance(nlags, int) or nlags < 0:
        raise ValueError("nlags must be a non-negative integer")
    nlags = min(nlags, nobs - 1)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")

    ac_values = acf(values, nlags=nlags, alpha=alpha).values
    pac_values = np.ones(nlags + 1, dtype=float)
    pac_values[0] = 1.0

    for k in range(1, nlags + 1):
        matrix = np.empty((k, k), dtype=float)
        for i in range(k):
            for j in range(k):
                matrix[i, j] = ac_values[abs(i - j)]
        rhs = ac_values[1 : k + 1]
        try:
            coefficients = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError("PACF calculation failed because the Yule-Walker system is singular") from exc
        pac_values[k] = coefficients[-1]

    z = _normal_critical_value(alpha)
    bound = z / np.sqrt(nobs)
    lags = np.arange(nlags + 1)
    lower = np.full(nlags + 1, -bound, dtype=float)
    upper = np.full(nlags + 1, bound, dtype=float)
    return PACFResult(lags, pac_values, lower, upper, nobs, name)


def _normal_critical_value(alpha: float) -> float:
    """Return the two-sided standard-normal critical value."""

    from scipy.stats import norm

    return float(norm.ppf(1.0 - alpha / 2.0))
