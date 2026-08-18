"""Workfile and sample management inspired by the EViews workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .series import TimeSeries


@dataclass
class Workfile:
    """Container for named time series and a current estimation sample."""

    series: dict[str, TimeSeries] = field(default_factory=dict)
    frequency: str | None = None
    sample_start: int | None = None
    sample_end: int | None = None

    @classmethod
    def from_dataframe(
        cls,
        frame: pd.DataFrame,
        *,
        index: Iterable | None = None,
        frequency: str | None = None,
    ) -> "Workfile":
        """Create a workfile from a numeric DataFrame."""
        idx = tuple(index if index is not None else frame.index)
        obj = cls(frequency=frequency)
        for column in frame.columns:
            obj.add(str(column), frame[column].to_numpy(dtype=float), index=idx)
        obj.sample_start = 0
        obj.sample_end = len(frame) - 1
        return obj

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        date_column: str | None = None,
        frequency: str | None = None,
        **kwargs,
    ) -> "Workfile":
        """Create a workfile from a CSV file."""
        frame = pd.read_csv(path, **kwargs)
        if date_column is not None:
            frame[date_column] = pd.to_datetime(frame[date_column])
            index = frame.pop(date_column)
        else:
            index = frame.index
        return cls.from_dataframe(frame, index=index, frequency=frequency)

    @property
    def nobs(self) -> int:
        """Return the number of observations in the workfile."""
        if not self.series:
            return 0
        return max(s.nobs for s in self.series.values())

    @property
    def sample(self) -> slice:
        """Return the current estimation sample as a Python slice."""
        start = 0 if self.sample_start is None else self.sample_start
        end = self.nobs if self.sample_end is None else self.sample_end + 1
        return slice(start, end)

    def add(
        self,
        name: str,
        values: TimeSeries | Iterable[float],
        *,
        index: Iterable | None = None,
    ) -> TimeSeries:
        """Add or replace a named series in the workfile."""
        if isinstance(values, TimeSeries):
            series = values.copy(name=name)
        else:
            series = TimeSeries(values, index=None if index is None else tuple(index), name=name, frequency=self.frequency)
        if self.series and series.nobs != self.nobs:
            raise ValueError("all workfile series must have the same number of positions")
        self.series[name] = series
        if self.sample_end is None:
            self.sample_end = series.nobs - 1
        return series

    def get(self, name: str) -> TimeSeries:
        """Return a named series."""
        try:
            return self.series[name]
        except KeyError as exc:
            raise KeyError(f"Series {name!r} is not present in the workfile") from exc

    def __getitem__(self, name: str) -> TimeSeries:
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self.series

    def names(self) -> list[str]:
        """Return series names in insertion order."""
        return list(self.series)

    def set_sample(self, start: int = 0, end: int | None = None) -> "Workfile":
        """Set the current estimation sample by positional bounds."""
        if self.nobs == 0:
            raise ValueError("cannot set a sample on an empty workfile")
        if start < 0 or start >= self.nobs:
            raise ValueError("sample start is outside the workfile")
        end = self.nobs - 1 if end is None else end
        if end < start or end >= self.nobs:
            raise ValueError("sample end is outside the workfile")
        self.sample_start = start
        self.sample_end = end
        return self

    def reset_sample(self) -> "Workfile":
        """Restore the full workfile as the current sample."""
        if self.nobs:
            self.sample_start = 0
            self.sample_end = self.nobs - 1
        return self

    def sample_series(self, name: str) -> TimeSeries:
        """Return the named series restricted to the current sample."""
        return self.get(name)[self.sample]

    def generate(self, name: str, expression, *, overwrite: bool = False) -> TimeSeries:
        """Generate a new series using a callable or simple expression callable.

        Expression may be a callable receiving this workfile. The callable can
        combine existing series and NumPy operations without introducing an
        EViews-specific parser into the numerical core.
        """
        if name in self.series and not overwrite:
            raise ValueError(f"series {name!r} already exists; set overwrite=True")
        values = expression(self) if callable(expression) else expression
        return self.add(name, values)

    def lag(self, name: str, periods: int = 1) -> TimeSeries:
        """Return a lagged workfile series."""
        return self.get(name).lag(periods)

    def diff(self, name: str, periods: int = 1) -> TimeSeries:
        """Return an ordinary difference of a workfile series."""
        return self.get(name).diff(periods)

    def describe(self) -> pd.DataFrame:
        """Return an EViews-style descriptive-statistics table for all series."""
        rows = []
        for name, series in self.series.items():
            stats = series.describe()
            stats["Series"] = name
            rows.append(stats)
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows).set_index("Series")
        return frame[["Observations", "Included observations", "Mean", "Std. Dev.", "Variance", "Minimum", "Maximum"]]

    def info(self) -> str:
        """Return workfile metadata similar to an EViews workfile view."""
        return "\n".join([
            "Workfile",
            "========",
            f"Frequency: {self.frequency or 'unspecified'}",
            f"Observations: {self.nobs}",
            f"Current sample: {self.sample_start if self.sample_start is not None else 0}"
            f" to {self.sample_end if self.sample_end is not None else max(self.nobs - 1, 0)}",
            f"Series: {', '.join(self.names()) if self.series else '(none)'}",
        ])
