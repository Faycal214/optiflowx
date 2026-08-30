"""Core time-series container used throughout StochX."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Sequence

import numpy as np
from scipy import stats

DateLike = date | datetime | str | int | float




def descriptive_statistics(values: Iterable[float]) -> dict[str, float]:
    """Compute EViews-style univariate descriptive statistics."""
    array = np.asarray(list(values), dtype=float).reshape(-1)
    if np.any(np.isinf(array)):
        raise ValueError("values must not contain infinite observations")
    valid = array[np.isfinite(array)]
    n = valid.size
    nan = float('nan')
    if n == 0:
        return {
            "Observations": float(array.size), "Included observations": 0.0,
            "Mean": nan, "Median": nan, "Std. Dev.": nan, "Variance": nan,
            "Minimum": nan, "Maximum": nan, "Skewness": nan, "Kurtosis": nan,
            "Jarque-Bera": nan, "Probability": nan,
        }
    mean = float(np.mean(valid))
    centered = valid - mean
    m2 = float(np.mean(centered ** 2))
    m3 = float(np.mean(centered ** 3))
    m4 = float(np.mean(centered ** 4))
    std = float(np.sqrt(np.sum(centered ** 2) / (n - 1))) if n > 1 else nan
    variance = float(np.sum(centered ** 2) / (n - 1)) if n > 1 else nan
    moment_std = float(np.sqrt(m2)) if m2 > 0 else 0.0
    skewness = float(m3 / moment_std**3) if n > 2 and moment_std > 0 else nan
    kurtosis = float(m4 / m2**2) if n > 3 and m2 > 0 else nan
    if np.isfinite(skewness) and np.isfinite(kurtosis):
        jb = float(n / 6.0 * (skewness**2 + ((kurtosis - 3.0) ** 2) / 4.0))
        jb_p = float(stats.chi2.sf(jb, 2))
    else:
        jb = jb_p = nan
    return {
        "Observations": float(array.size), "Included observations": float(n),
        "Mean": mean, "Median": float(np.median(valid)),
        "Std. Dev.": std, "Variance": variance,
        "Minimum": float(np.min(valid)), "Maximum": float(np.max(valid)),
        "Skewness": skewness, "Kurtosis": kurtosis,
        "Jarque-Bera": jb, "Probability": jb_p,
    }
@dataclass(frozen=True)
class TimeSeries:
    """One regularly spaced univariate time series.

    Parameters
    ----------
    values:
        Numeric observations in chronological order. Missing observations may
        be represented by ``numpy.nan``.
    index:
        Optional labels for observations. When omitted, integer positions
        starting at zero are used.
    name:
        Optional series name, used in summaries and plots.
    frequency:
        Optional frequency label such as ``"A"``, ``"Q"``, ``"M"`` or ``"D"``.

    Notes
    -----
    The course material focuses on regularly spaced observations. The
    container therefore preserves order and frequency metadata without trying
    to infer or repair irregular calendars automatically.
    """

    values: np.ndarray
    index: tuple[DateLike, ...] | None = None
    name: str = "series"
    frequency: str | None = None

    def __init__(self, values: Iterable[float], index: Sequence[DateLike] | None = None, *, name: str = "series", frequency: str | None = None) -> None:
        array = np.asarray(list(values), dtype=float)
        if array.ndim != 1:
            raise ValueError("values must be a one-dimensional numeric sequence")
        if len(array) == 0:
            raise ValueError("values must contain at least one observation")
        if np.any(np.isinf(array)):
            raise ValueError("values must not contain infinite observations")
        labels = None if index is None else tuple(index)
        if labels is not None and len(labels) != len(array):
            raise ValueError("index and values must have the same length")
        object.__setattr__(self, "values", array.copy())
        object.__setattr__(self, "index", labels)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "frequency", frequency)

    def __len__(self) -> int:
        return self.values.size

    def __getitem__(self, item: int | slice | np.ndarray | list[bool]) -> float | "TimeSeries":
        if isinstance(item, (slice, np.ndarray, list)):
            values = self.values[item]
            if self.index is None:
                index = None
            elif isinstance(item, slice):
                index = self.index[item]
            else:
                mask = np.asarray(item)
                index = tuple(label for label, keep in zip(self.index, mask) if bool(keep))
            return TimeSeries(values, index=index, name=self.name, frequency=self.frequency)
        return float(self.values[item])

    @property
    def nobs(self) -> int:
        """Number of positions in the series, including missing observations."""
        return len(self)

    @property
    def nmissing(self) -> int:
        """Number of missing observations."""
        return int(np.isnan(self.values).sum())

    @property
    def start(self) -> DateLike | None:
        """First index label, when an index was provided."""
        return None if self.index is None else self.index[0]

    @property
    def end(self) -> DateLike | None:
        """Last index label, when an index was provided."""
        return None if self.index is None else self.index[-1]

    def copy(self, *, name: str | None = None) -> "TimeSeries":
        """Return an independent copy of the series."""
        return TimeSeries(self.values.copy(), index=self.index, name=self.name if name is None else name, frequency=self.frequency)

    def lag(self, periods: int = 1) -> "TimeSeries":
        """Return a lagged series, padding unavailable observations with NaN."""
        if not isinstance(periods, int):
            raise TypeError("periods must be an integer")
        result = np.full(self.nobs, np.nan, dtype=float)
        if periods >= 0:
            if periods < self.nobs:
                result[periods:] = self.values[: self.nobs - periods]
        else:
            shift = -periods
            if shift < self.nobs:
                result[: self.nobs - shift] = self.values[shift:]
        return TimeSeries(result, index=self.index, name=f"{self.name}_lag{periods}", frequency=self.frequency)

    def diff(self, periods: int = 1) -> "TimeSeries":
        """Return the EViews-style difference, preserving workfile length."""
        if not isinstance(periods, int) or periods < 1:
            raise ValueError("periods must be a positive integer")
        if self.nobs <= periods:
            raise ValueError("periods must be smaller than the number of observations")
        values = self.values.copy()
        for _ in range(periods):
            differenced = np.full(values.size, np.nan, dtype=float)
            differenced[1:] = values[1:] - values[:-1]
            values = differenced
        return TimeSeries(
            values,
            index=self.index,
            name=f"D({self.name},{periods})",
            frequency=self.frequency,
        )
    def log(self) -> "TimeSeries":
        """Return the natural logarithm of a strictly positive series."""
        if np.isnan(self.values).any():
            raise ValueError("log requires a series without missing observations")
        if np.any(self.values <= 0):
            raise ValueError("log transformation requires strictly positive values")
        return TimeSeries(np.log(self.values), index=self.index, name=f"LOG({self.name})", frequency=self.frequency)

    def pct_change(self, periods: int = 1) -> "TimeSeries":
        """Return percentage changes over ``periods`` observations."""
        if not isinstance(periods, int) or periods < 1:
            raise ValueError("periods must be a positive integer")
        if periods >= self.nobs:
            raise ValueError("periods must be smaller than the number of observations")
        if np.isnan(self.values).any():
            raise ValueError("pct_change requires a series without missing observations")
        previous = self.values[:-periods]
        current = self.values[periods:]
        if np.any(previous == 0):
            raise ZeroDivisionError("percentage change is undefined after a zero")
        values = (current / previous - 1.0) * 100.0
        index = self.index[periods:] if self.index is not None else None
        return TimeSeries(values, index=index, name=f"DLOG({self.name})", frequency=self.frequency)

    def describe(self) -> dict[str, float]:
        """Return EViews-compatible descriptive statistics for this series.

        The statistics use all non-missing observations in the series.
        Workfile-level reports apply the active sample before calculation.
        """
        return descriptive_statistics(self.values)
    def stats(self) -> dict[str, float]:
        """Alias for the EViews-style descriptive-statistics view."""
        return self.describe()

    def summary(self) -> str:
        """Return a compact EViews-style descriptive summary."""
        s = self.describe()
        lines = [self.name, "=" * len(self.name)]
        if self.frequency is not None:
            lines.append(f"Frequency:             {self.frequency}")
        for key in ["Observations", "Included observations", "Mean", "Median", "Maximum", "Minimum", "Std. Dev.", "Skewness", "Kurtosis", "Jarque-Bera", "Probability"]:
            value = s[key]
            display = str(int(value)) if key in {"Observations", "Included observations"} else f"{value:.6f}"
            lines.append(f"{key:24s} {display}")
        return "\n".join(lines)

    def acf(self, nlags: int | None = None, *, alpha: float = 0.05):
        """Estimate and return the sample autocorrelation function."""
        from .correlation import acf
        return acf(self, nlags=nlags, alpha=alpha)

    def pacf(self, nlags: int | None = None, *, alpha: float = 0.05):
        """Estimate and return the sample partial autocorrelation function."""
        from .correlation import pacf
        return pacf(self, nlags=nlags, alpha=alpha)

    def __repr__(self) -> str:
        return f"TimeSeries(name={self.name!r}, nobs={self.nobs}, frequency={self.frequency!r})"
