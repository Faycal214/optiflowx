"""Stage 9.6 Box-Jenkins forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .box_jenkins_selection import BoxJenkinsSelectionResult
from .forecasting import restore_differences


@dataclass(frozen=True)
class BoxJenkinsForecastResult:
    """Auditable point and interval forecasts from the selected model."""

    order: tuple[int, int, int]
    criterion: str
    forecast_horizon: int
    alpha: float
    index: pd.Index
    forecast: np.ndarray
    standard_error: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    scale: str
    restoration_order: int
    restored_from_differences: bool
    restoration_history_nobs: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "order", tuple(int(v) for v in self.order))
        object.__setattr__(self, "forecast_horizon", int(self.forecast_horizon))
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "index", pd.Index(self.index).copy())
        for name in ("forecast", "standard_error", "lower", "upper"):
            values = np.asarray(getattr(self, name), dtype=float).copy()
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        object.__setattr__(self, "restoration_order", int(self.restoration_order))
        if self.restoration_history_nobs is not None:
            object.__setattr__(self, "restoration_history_nobs", int(self.restoration_history_nobs))
        if len(self.index) != self.forecast_horizon:
            raise ValueError("forecast index length must equal forecast_horizon")
        expected = (self.forecast_horizon,)
        for name in ("forecast", "standard_error", "lower", "upper"):
            if getattr(self, name).shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
        if np.any(np.isfinite(self.lower) & np.isfinite(self.forecast) & (self.lower > self.forecast)):
            raise ValueError("lower forecast bounds must not exceed the point forecast")
        if np.any(np.isfinite(self.upper) & np.isfinite(self.forecast) & (self.upper < self.forecast)):
            raise ValueError("upper forecast bounds must not be below the point forecast")

    @property
    def point_forecast(self) -> np.ndarray:
        return self.forecast

    @property
    def confidence_level(self) -> float:
        return 1.0 - self.alpha

    @property
    def horizon(self) -> int:
        return self.forecast_horizon

    @property
    def restored(self) -> bool:
        return self.restored_from_differences

    def table(self) -> pd.DataFrame:
        """Return the deterministic forecast table."""
        return pd.DataFrame(
            {
                "Forecast": self.forecast,
                "Std. Error": self.standard_error,
                "Lower": self.lower,
                "Upper": self.upper,
            },
            index=self.index,
        )

    def metadata(self) -> dict[str, object]:
        """Return auditable forecast metadata."""
        return {
            "order": self.order,
            "criterion": self.criterion,
            "forecast_horizon": self.forecast_horizon,
            "alpha": self.alpha,
            "confidence_level": self.confidence_level,
            "scale": self.scale,
            "restoration_order": self.restoration_order,
            "restored_from_differences": self.restored_from_differences,
            "restoration_history_nobs": self.restoration_history_nobs,
        }


def _next_index(original: object, steps: int) -> pd.Index:
    if isinstance(original, pd.Series):
        idx = original.index
    elif hasattr(original, "index"):
        idx = pd.Index(original.index)
    else:
        idx = pd.RangeIndex(0, 0)
    if isinstance(idx, pd.DatetimeIndex):
        freq = idx.freq or idx.inferred_freq
        if freq is not None and len(idx):
            return pd.date_range(start=idx[-1] + freq, periods=steps, freq=freq)
    if isinstance(idx, pd.PeriodIndex):
        return pd.period_range(start=idx[-1] + 1, periods=steps, freq=idx.freq)
    if len(idx):
        try:
            return pd.Index(range(int(idx[-1]) + 1, int(idx[-1]) + 1 + steps))
        except (TypeError, ValueError):
            pass
    return pd.RangeIndex(1, steps + 1)


def _forecast_frame(selected, steps: int, alpha: float) -> pd.DataFrame:
    if selected.ts_result is None:
        raise ValueError("selected candidate has no fitted model")
    frame = selected.ts_result.forecast(steps=steps, alpha=alpha)
    if not {"Forecast", "Lower", "Upper"}.issubset(frame.columns):
        raise ValueError("fitted-model forecast must provide Forecast, Lower, and Upper")
    if "Std. Error" not in frame.columns:
        if "mean_se" in frame.columns:
            frame["Std. Error"] = frame["mean_se"]
        else:
            frame["Std. Error"] = np.nan
    return frame[["Forecast", "Std. Error", "Lower", "Upper"]]


def forecast_box_jenkins(
    selection: BoxJenkinsSelectionResult,
    *,
    steps: int,
    alpha: float = 0.05,
    forecast_on_differenced_scale: bool = False,
    last_levels: Iterable[float] | None = None,
    restoration_order: int | None = None,
    forecast_index: pd.Index | Iterable[object] | None = None,
) -> BoxJenkinsForecastResult:
    """Forecast from the selected adequate Stage 9.5 model.

    The existing ARIMA estimator returns forecasts on the original modeled
    scale. Inverse differencing is therefore opt-in for callers that explicitly
    provide forecasts on a differenced scale.
    """
    if not selection.has_selection or selection.selected is None:
        raise ValueError("forecasting requires a selected adequate model")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not 0 < float(alpha) < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")

    selected = selection.selected
    frame = _forecast_frame(selected, steps, float(alpha))
    original = getattr(selected.ts_result, "original", None)
    if forecast_index is not None:
        index = pd.Index(forecast_index)
        if len(index) != steps:
            raise ValueError("forecast_index length must equal steps")
    else:
        # The fitted-model forecast index may be a positional RangeIndex even
        # when the original series carries a meaningful DatetimeIndex or
        # PeriodIndex.  The Stage 9 contract requires the forecast index to be
        # derived from the original model input unless the caller overrides it.
        index = _next_index(original, steps)

    point = frame["Forecast"].to_numpy(dtype=float)
    se = frame["Std. Error"].to_numpy(dtype=float)
    lower = frame["Lower"].to_numpy(dtype=float)
    upper = frame["Upper"].to_numpy(dtype=float)
    restored = False
    order = 0
    history_nobs: int | None = None
    scale = "original"

    if forecast_on_differenced_scale:
        if last_levels is None:
            raise ValueError("last_levels are required when forecasting on a differenced scale")
        order = selected.order[1] if restoration_order is None else int(restoration_order)
        if order < 0:
            raise ValueError("restoration_order must be non-negative")
        history = np.asarray(list(last_levels), dtype=float)
        if history.size < 1 or not np.isfinite(history).all():
            raise ValueError("last_levels must contain at least one finite observation")
        point = restore_differences(history, point, order=order)
        lower = restore_differences(history, lower, order=order)
        upper = restore_differences(history, upper, order=order)
        history_nobs = int(history.size)
        restored = order > 0
        scale = "original" if restored else "differenced"

    return BoxJenkinsForecastResult(
        order=selected.order,
        criterion=selection.criterion,
        forecast_horizon=steps,
        alpha=float(alpha),
        index=index,
        forecast=point,
        standard_error=se,
        lower=lower,
        upper=upper,
        scale=scale,
        restoration_order=order,
        restored_from_differences=restored,
        restoration_history_nobs=history_nobs,
    )
