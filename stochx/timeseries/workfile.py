"""Workfile and sample management inspired by the EViews workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

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

    def set_sample(self, start: int | str = 0, end: int | str | None = None) -> "Workfile":
        """Set the current estimation sample by positions or matching index labels."""
        if self.nobs == 0:
            raise ValueError("cannot set a sample on an empty workfile")
        if isinstance(start, str):
            parts = start.split()
            if len(parts) not in {1, 2}:
                raise ValueError("sample string must be 'start' or 'start end'")
            start_label = parts[0]
            end_label = parts[1] if len(parts) == 2 else parts[0]
            reference = next(iter(self.series.values()))
            if reference.index is None:
                raise ValueError("string samples require an indexed workfile")
            labels = list(reference.index)
            start_pos = next((i for i, value in enumerate(labels) if str(value) == start_label), -1)
            end_pos = next((i for i, value in enumerate(labels) if str(value) == end_label), -1)
            if start_pos < 0 or end_pos < 0:
                try:
                    parsed = pd.to_datetime(pd.Index(labels))
                    start_pos = int(np.where(parsed == pd.Timestamp(start_label))[0][0])
                    end_pos = int(np.where(parsed == pd.Timestamp(end_label))[0][0])
                except Exception as exc:  # noqa: BLE001
                    raise ValueError(f"could not resolve sample labels {start_label!r}, {end_label!r}") from exc
            start, end = start_pos, end_pos
        if not isinstance(start, int):
            raise TypeError("sample start must be an integer position or label string")
        if start < 0 or start >= self.nobs:
            raise ValueError("sample start is outside the workfile")
        if end is None:
            end = self.nobs - 1
        if not isinstance(end, int):
            raise TypeError("sample end must be an integer position or label string")
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

    def eval(self, expression: str):
        """Evaluate an EViews-inspired time-series expression."""
        from .expression import evaluate

        return evaluate(expression, self)

    def _pad_to_workfile(self, value: TimeSeries, *, name: str) -> TimeSeries:
        """Align a shorter expression result to the workfile with leading NaNs."""
        if value.nobs == self.nobs:
            return value.copy(name=name)
        if value.nobs > self.nobs:
            raise ValueError("expression result is longer than the workfile")
        pad = self.nobs - value.nobs
        values = np.r_[np.full(pad, np.nan), value.values]
        if self.series:
            reference = next(iter(self.series.values()))
            index = reference.index
        else:
            index = None
        return TimeSeries(values, index=index, name=name, frequency=self.frequency)

    def generate(self, name: str, expression, *, overwrite: bool = False) -> TimeSeries:
        """Generate a new series using an expression string or callable."""
        if name in self.series and not overwrite:
            raise ValueError(f"series {name!r} already exists; set overwrite=True")
        if isinstance(expression, str):
            values = self.eval(expression)
        elif callable(expression):
            values = expression(self)
        else:
            values = expression
        if not isinstance(values, TimeSeries):
            if not self.series:
                raise ValueError("a generated scalar requires an existing reference series")
            values = TimeSeries(np.full(self.nobs, float(values)), name=name, frequency=self.frequency)
        return self.add(name, self._pad_to_workfile(values, name=name))

    def series_from_expression(self, name: str, expression: str, *, overwrite: bool = False) -> TimeSeries:
        """Convenience alias for EViews-style ``series name = expression`` generation."""
        return self.generate(name, expression, overwrite=overwrite)

    def lag(self, name: str, periods: int = 1) -> TimeSeries:
        """Return a lagged workfile series."""
        return self.get(name).lag(periods)

    def diff(self, name: str, periods: int = 1) -> TimeSeries:
        """Return an ordinary difference of a workfile series."""
        return self.get(name).diff(periods)

    def equation(self, name: str = "EQ01", specification: str = ""):
        """Create an equation object attached to this workfile."""
        from .equation import Equation

        return Equation(self, name=name, specification=specification)

    def ls(self, specification: str, *, name: str = "EQ01"):
        """Estimate an OLS equation using EViews-like ``Y C X(-1)`` syntax."""
        return self.equation(name, specification).ls()

    def estimate(self, specification: str, *, method: str = "LS", name: str = "EQ01"):
        """Estimate an equation using the supported EViews-style estimator."""
        return self.equation(name, specification).estimate(method=method)

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
