"""Workfile and sample management inspired by the EViews workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Iterable
import re

import numpy as np
import pandas as pd

from .series import TimeSeries, descriptive_statistics


@dataclass
class Workfile:
    """Container for named time series and a current estimation sample."""

    series: dict[str, TimeSeries] = field(default_factory=dict)
    frequency: str | None = None
    sample_start: int | None = None
    sample_end: int | None = None
    sample_mask: np.ndarray | None = None

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

    _RESERVED_NAMES: ClassVar[frozenset[str]] = frozenset({
        "ABS", "ACOS", "AND", "AR", "ASIN", "C", "CON", "CNORM", "COEF",
        "COS", "D", "DLOG", "DNORM", "ELSE", "ENDIF", "EXP", "LOG", "LOGIT",
        "MA", "NA", "NOT", "NRND", "OR", "PDL", "RESID", "RND", "SAR", "SIN",
        "SMA", "SQR", "THEN",
    })

    @staticmethod
    def _name_key(name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("series name must be a non-empty string")
        return name.strip().casefold()

    def add(
        self,
        name: str,
        values: TimeSeries | Iterable[float],
        *,
        index: Iterable | None = None,
    ) -> TimeSeries:
        """Add or replace a named series in the workfile.

        EViews object names are case-insensitive; the original capitalization
        is retained for display and reporting.
        """
        name = name.strip()
        key = self._name_key(name)
        if key.upper() in self._RESERVED_NAMES:
            raise ValueError(f"series name {name!r} is reserved by EViews")
        existing_key = next((k for k in self.series if self._name_key(k) == key), None)
        if existing_key is not None and existing_key != name:
            del self.series[existing_key]
        if isinstance(values, TimeSeries):
            series = values.copy(name=name)
        else:
            resolved_index = index
            if resolved_index is None and self.series:
                resolved_index = next(iter(self.series.values())).index
            series = TimeSeries(
                values,
                index=None if resolved_index is None else tuple(resolved_index),
                name=name,
                frequency=self.frequency,
            )
        if self.series:
            reference = next(iter(self.series.values()))
            if series.nobs != self.nobs:
                raise ValueError("all workfile series must have the same number of observations")
            if reference.index is not None and series.index is not None and reference.index != series.index:
                raise ValueError("all indexed workfile series must share the same index")
            if reference.index is not None and series.index is None:
                series = series.copy(name=name)
                object.__setattr__(series, "index", reference.index)
            if self.frequency is not None and series.frequency is not None and series.frequency != self.frequency:
                raise ValueError("all workfile series must share the workfile frequency")
        self.series[name] = series
        if self.sample_end is None:
            self.sample_end = series.nobs - 1
        return series

    def get(self, name: str) -> TimeSeries:
        """Return a named series using EViews-compatible case-insensitive lookup."""
        key = self._name_key(name)
        for existing, series in self.series.items():
            if self._name_key(existing) == key:
                return series
        raise KeyError(f"Series {name!r} is not present in the workfile")

    def __getitem__(self, name: str) -> TimeSeries:
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        key = self._name_key(name)
        return any(self._name_key(existing) == key for existing in self.series)

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

            def _resolve(label: str) -> int:
                matches = [i for i, value in enumerate(labels) if str(value) == label]
                if matches:
                    return matches[0]
                try:
                    parsed = pd.to_datetime(pd.Index(labels))
                    stamp = pd.Timestamp(label)
                    matches = np.where(parsed == stamp)[0]
                    if matches.size:
                        return int(matches[0])
                except Exception:
                    pass
                # EViews-style period labels: YYYYQn and YYYYMm.
                period_match = re.fullmatch(r"(\d{4})([QM])(\d{1,2})", label.upper())
                if period_match:
                    year = int(period_match.group(1))
                    unit = period_match.group(2)
                    period = int(period_match.group(3))
                    if unit == "Q" and not 1 <= period <= 4:
                        raise ValueError(f"invalid quarterly sample label {label!r}")
                    if unit == "M" and not 1 <= period <= 12:
                        raise ValueError(f"invalid monthly sample label {label!r}")
                    try:
                        target = pd.Period(
                            f"{year}Q{period}" if unit == "Q" else f"{year}-{period:02d}",
                            freq="Q" if unit == "Q" else "M",
                        )
                        for i, value in enumerate(labels):
                            if isinstance(value, pd.Period) and value == target:
                                return i
                            try:
                                if pd.Period(pd.Timestamp(value), freq=target.freq) == target:
                                    return i
                            except Exception:
                                pass
                    except Exception:
                        pass
                raise ValueError(f"could not resolve sample label {label!r}")

            start, end = _resolve(start_label), _resolve(end_label)
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
        self.sample_mask = None
        return self

    def reset_sample(self) -> "Workfile":
        """Restore the full workfile as the current sample."""
        if self.nobs:
            self.sample_start = 0
            self.sample_end = self.nobs - 1
            self.sample_mask = None
        return self

    @property
    def sample_indexer(self):
        """Return the active sample as a slice or boolean mask."""
        if self.sample_mask is not None:
            return self.sample_mask
        return self.sample

    def sample_series(self, name: str) -> TimeSeries:
        """Return the named series restricted to the current sample."""
        return self.get(name)[self.sample_indexer]

    def smpl(self, specification: str) -> "Workfile":
        """Set the current workfile sample using EViews-style syntax."""
        if not isinstance(specification, str) or not specification.strip():
            raise ValueError("sample specification must be a non-empty string")
        text = specification.strip()
        if text.lower().startswith("smpl "):
            text = text[5:].strip()

        match = re.search(r"\s+if\s+", text, flags=re.IGNORECASE)
        condition = None
        base = text
        if match:
            base = text[:match.start()].strip()
            condition = text[match.end():].strip()
            if not condition:
                raise ValueError("sample condition after 'if' must not be empty")

        if not base or base.lower() == "@all":
            self.reset_sample()
        else:
            self.set_sample(base)

        if condition is not None:
            from .expression import ExpressionError, evaluate
            try:
                result = evaluate(condition, self)
            except ExpressionError as exc:
                raise ValueError(f"invalid sample condition: {exc}") from exc
            if not isinstance(result, TimeSeries):
                raise ValueError("sample condition must evaluate to a series")
            values = np.asarray(result.values, dtype=float)
            mask = np.isfinite(values) & (values != 0)
            base_mask = np.zeros(self.nobs, dtype=bool)
            base_mask[self.sample] = True
            self.sample_mask = base_mask & mask
        return self

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
        exists = name in self
        if exists and not overwrite:
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

    def correlogram(
        self,
        name: str,
        *,
        nlags: int,
        d: int = 0,
        model_df: int = 0,
        alpha: float = 0.05,
    ):
        """Compute an EViews-style correlogram on the active sample."""
        from .correlogram import correlogram
        return correlogram(
            self.sample_series(name),
            nlags=nlags,
            d=d,
            model_df=model_df,
            alpha=alpha,
        )

    def acf(self, name: str, *, nlags: int | None = None, d: int = 0, alpha: float = 0.05):
        """Compute the ACF of a workfile series on the active sample."""
        from .correlation import acf
        series = self.sample_series(name)
        if d:
            series = series.diff(d)
        return acf(series, nlags=nlags, alpha=alpha)

    def pacf(self, name: str, *, nlags: int | None = None, d: int = 0, alpha: float = 0.05):
        """Compute the PACF of a workfile series on the active sample."""
        from .correlation import pacf
        series = self.sample_series(name)
        if d:
            series = series.diff(d)
        return pacf(series, nlags=nlags, alpha=alpha)

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

    def describe(self, *, individual: bool = False) -> pd.DataFrame:
        """Return EViews-style descriptive statistics for workfile series.

        By default, all series use a common sample after removing rows with
        any missing value. Set individual=True for per-series samples.
        """
        active = self.sample_indexer
        if not self.series:
            return pd.DataFrame()
        if individual:
            samples = {name: series[active].values for name, series in self.series.items()}
        else:
            arrays = [series[active].values for series in self.series.values()]
            common = np.ones(len(arrays[0]), dtype=bool)
            for values in arrays:
                common &= np.isfinite(values)
            samples = {name: series[active].values[common] for name, series in self.series.items()}
        rows = []
        for name, values in samples.items():
            row = descriptive_statistics(values)
            row["Series"] = name
            rows.append(row)
        frame = pd.DataFrame(rows).set_index("Series")
        return frame[[
            "Observations", "Included observations", "Mean", "Std. Dev.",
            "Variance", "Minimum", "Maximum", "Skewness", "Kurtosis",
            "Jarque-Bera", "Probability",
        ]]
    def stats(self, *, individual: bool = False) -> pd.DataFrame:
        """Alias for the EViews-style workfile descriptive-statistics view."""
        return self.describe(individual=individual)

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
