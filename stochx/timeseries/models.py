"""AR, MA, ARMA, ARIMA and SARIMA estimation for the course workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .results import UnifiedResult
from .series import TimeSeries


@dataclass
class TSResult(UnifiedResult):
    """Unified StochX result wrapper around a fitted time-series model."""

    fitted_model: Any = None
    original: TimeSeries | pd.Series | None = None
    order: tuple[int, int, int] | None = None
    seasonal_order: tuple[int, int, int, int] | None = None

    @property
    def model_name(self) -> str:
        """Return the model family name."""
        return self.title

    def forecast(self, steps: int = 1, alpha: float = 0.05) -> pd.DataFrame:
        """Forecast future values with prediction intervals.

        Statsmodels exposes ``get_forecast`` for ARIMA/SARIMAX-style results,
        while AutoReg exposes ``get_prediction``. Normalize both paths to the
        same summary-frame contract without changing either estimator.
        """
        if steps < 1:
            raise ValueError("steps must be positive")

        if hasattr(self.result, "get_forecast"):
            prediction = self.result.get_forecast(steps=steps)
        elif hasattr(self.result, "get_prediction"):
            # AutoRegResults does not implement get_forecast; its prediction
            # API accepts explicit start/end positions for future observations.
            nobs = int(getattr(self.result, "nobs", len(self.original) if self.original is not None else 0))
            prediction = self.result.get_prediction(start=nobs, end=nobs + steps - 1, dynamic=False)
        else:
            raise AttributeError("fitted model does not provide a forecast API")

        frame = prediction.summary_frame(alpha=alpha)
        return frame.rename(
            columns={
                "mean": "Forecast",
                "mean_ci_lower": "Lower",
                "mean_ci_upper": "Upper",
            }
        )

    def diagnostics(self, lags: int = 12):
        """Run the standard StochX residual validation battery."""
        from .diagnostics import residual_diagnostics

        return residual_diagnostics(
            self.residuals,
            lags=lags,
            p=self.order[0] if self.order else 0,
            q=self.order[2] if self.order else 0,
        )

    def residual_correlogram(self, lags: int = 12, *, alpha: float = 0.05):
        """Return the frozen EViews-style correlogram of model residuals."""
        from .diagnostics import residual_diagnostics_correlogram

        return residual_diagnostics_correlogram(self, lags=lags, alpha=alpha)

    def correlogram(self, lags: int = 12, *, alpha: float = 0.05):
        """Alias for :meth:`residual_correlogram` in the diagnostics workflow."""
        return self.residual_correlogram(lags=lags, alpha=alpha)

    def roots(self) -> dict[str, np.ndarray]:
        """Return AR and MA roots when available."""
        ar_roots = np.asarray(getattr(self.result, "arroots", []), dtype=complex)
        ma_roots = np.asarray(getattr(self.result, "maroots", []), dtype=complex)
        return {"AR roots": ar_roots, "MA roots": ma_roots}

    def stability(self) -> dict[str, bool]:
        """Check the course stationarity and invertibility root conditions."""
        roots = self.roots()
        ar_ok = bool(np.all(np.abs(roots["AR roots"]) > 1.0)) if roots["AR roots"].size else True
        ma_ok = bool(np.all(np.abs(roots["MA roots"]) > 1.0)) if roots["MA roots"].size else True
        return {"stationary": ar_ok, "invertible": ma_ok}

    def interpret(self, alpha: float = 0.05) -> str:
        """Interpret coefficient significance and time-series model adequacy."""
        base = super().interpret(alpha=alpha)
        stability = self.stability()
        root_statement = f"Stability check: stationary={stability['stationary']}, invertible={stability['invertible']}."
        return f"{base} {root_statement}"


def _as_series(y: TimeSeries | Iterable[float]) -> pd.Series:
    if isinstance(y, TimeSeries):
        index = y.index if y.index is not None else None
        return pd.Series(y.values, index=index, name=y.name, dtype=float)
    return pd.Series(np.asarray(list(y), dtype=float), dtype=float)


def _result(model_name: str, model: Any, result: Any, y, order, seasonal_order=None) -> TSResult:
    return TSResult(
        result=result,
        title=model_name,
        dependent=getattr(y, "name", "Y"),
        method="Maximum Likelihood" if model_name != "AR" else "Conditional Least Squares / Yule-Walker-compatible AR estimation",
        fitted_model=model,
        original=y,
        order=order,
        seasonal_order=seasonal_order,
    )


def fit_ar(y: TimeSeries | Iterable[float], p: int, *, trend: str = "c", method: str = "yule_walker") -> TSResult:
    """Estimate AR(p); the course treats Yule-Walker/OLS as the AR-specific route."""
    if p < 1:
        raise ValueError("p must be positive")
    series = _as_series(y)
    from statsmodels.tsa.ar_model import AutoReg

    trend_map = {"n": "n", "c": "c", "ct": "ct"}
    if trend not in trend_map:
        raise ValueError("trend must be 'n', 'c', or 'ct'")
    model = AutoReg(series, lags=p, trend=trend_map[trend], old_names=False)
    result = model.fit()
    return _result("AR", model, result, y, (p, 0, 0))


def fit_ma(y: TimeSeries | Iterable[float], q: int, *, trend: str = "c") -> TSResult:
    """Estimate MA(q) under Gaussian maximum likelihood."""
    if q < 1:
        raise ValueError("q must be positive")
    series = _as_series(y)
    from statsmodels.tsa.arima.model import ARIMA
