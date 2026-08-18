"""Core time-series container used throughout StochX."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from typing import Iterable, Sequence

import numpy as np


DateLike = date | datetime | str | int | float


@dataclass(frozen=True)
class TimeSeries:
    """One regularly spaced univariate time series.

    Parameters
    ----------
    values:
        Numeric observations in chronological order.
    index:
        Optional labels for the observations. When omitted, integer positions
        starting at zero are used.
    name:
        Optional series name, used in summaries and plots.
    frequency:
        Optional frequency label such as ``"A"``, ``"Q"``, ``"M"`` or ``"D"``.

    Notes
    -----
    The current course material focuses on regularly spaced observations. The
    container therefore preserves order and frequency metadata without trying
    to infer or repair irregular calendars automatically.
    """

    values: np.ndarray
    index: tuple[DateLike, ...] | None = None
    name: str = "series"
    frequency: str | None = None

    def __init__(
        self,
        values: Iterable[float],
        index: Sequence[DateLike] | None = None,
        *,
        name: str = "series",
        frequency: str | None = None,
    ) -> None:
        array = np.asarray(list(values), dtype=float)
        if array.ndim != 1:
            raise ValueError("values must be a one-dimensional numeric sequence")
        if len(array) == 0:
            raise ValueError("values must contain at least one observation")
        if not np.all(np.isfinite(array)):
            raise ValueError("values must contain only finite observations")

        if index is not None:
            labels = tuple(index)
            if len(labels) != len(array):
                raise ValueError("index and values must have the same length")
        else:
            labels = None

        object.__setattr__(self, "values", array.copy())
        object.__setattr__(self, "index", labels)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "frequency", frequency)

    def __len__(self) -> int:
        return self.values.size

    def __getitem__(self, item: int | slice) -> float | "TimeSeries":
        if isinstance(item, slice):
            values = self.values[item]
            index = self.index[item] if self.index is not None else None
            return TimeSeries(
                values,
                index=index,
                name=self.name,
                frequency=self.frequency,
            )
        return float(self.values[item])

    @property
    def nobs(self) -> int:
        """Number of observations."""

        return len(self)

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

        return TimeSeries(
            self.values.copy(),
            index=self.index,
            name=self.name if name is None else name,
            frequency=self.frequency,
        )

    def lag(self, periods: int = 1) -> "TimeSeries":
        """Return a lagged series, padding unavailable observations with NaN.

        Positive ``periods`` means ``y[t-periods]``. This mirrors the usual
        time-series lag convention and is intended for model construction.
        """

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
        return TimeSeries(
            result,
            index=self.index,
            name=f"{self.name}_lag{periods}",
            frequency=self.frequency,
        )

    def diff(self, periods: int = 1) -> "TimeSeries":
        """Difference the series ``periods`` times."""

        if not isinstance(periods, int) or periods < 1:
            raise ValueError("periods must be a positive integer")
        values = self.values.copy()
        for _ in range(periods):
            values = np.diff(values)
        index = self.index[periods:] if self.index is not None else None
        return TimeSeries(
            values,
            index=index,
            name=f"D({self.name},{periods})",
            frequency=self.frequency,
        )

    def log(self) -> "TimeSeries":
        """Return the natural logarithm of a strictly positive series."""

        if np.any(self.values <= 0):
            raise ValueError("log transformation requires strictly positive values")
        return TimeSeries(
            np.log(self.values),
            index=self.index,
            name=f"LOG({self.name})",
            frequency=self.frequency,
        )

    def pct_change(self, periods: int = 1) -> "TimeSeries":
        """Return percentage changes over ``periods`` observations."""

        if not isinstance(periods, int) or periods < 1:
            raise ValueError("periods must be a positive integer")
        if periods >= self.nobs:
            raise ValueError("periods must be smaller than the number of observations")
        previous = self.values[:-periods]
        current = self.values[periods:]
        if np.any(previous == 0):
            raise ZeroDivisionError("percentage change is undefined after a zero")
        values = (current / previous - 1.0) * 100.0
        index = self.index[periods:] if self.index is not None else None
        return TimeSeries(
            values,
            index=index,
            name=f"DLOG({self.name})",
            frequency=self.frequency,
        )

    def describe(self) -> dict[str, float]:
        """Return the core descriptive statistics used in course TPs."""

        return {
            "Observations": float(self.nobs),
            "Mean": float(np.mean(self.values)),
            "Std. Dev.": float(np.std(self.values, ddof=1)) if self.nobs > 1 else float("nan"),
            "Variance": float(np.var(self.values, ddof=1)) if self.nobs > 1 else float("nan"),
            "Minimum": float(np.min(self.values)),
            "Maximum": float(np.max(self.values)),
        }

    def summary(self) -> str:
        """Return a compact EViews-style descriptive summary."""

        stats = self.describe()
        lines = [
            f"{self.name}",
            "=" * len(self.name),
            f"Observations: {int(stats['Observations'])}",
            f"Mean:        {stats['Mean']:.6g}",
            f"Std. Dev.:   {stats['Std. Dev.']:.6g}",
            f"Variance:    {stats['Variance']:.6g}",
            f"Minimum:     {stats['Minimum']:.6g}",
            f"Maximum:     {stats['Maximum']:.6g}",
        ]
        return "\n".join(lines)

    def _replace_nan(self, values: np.ndarray) -> "TimeSeries":
        return TimeSeries(values, index=self.index, name=self.name, frequency=self.frequency)

    def __repr__(self) -> str:
        return (
            f"TimeSeries(name={self.name!r}, nobs={self.nobs}, "
            f"frequency={self.frequency!r})"
        )
