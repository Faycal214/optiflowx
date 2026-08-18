"""AR, MA, ARMA, ARIMA and SARIMA estimation for the course workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .series import TimeSeries


def _as_series(y: TimeSeries | Iterable[float]) -> pd.Series:
    if isinstance(y, TimeSeries):
        index = y.index if y.index is not None else None
        return pd.Series(y.values, index=index, name=y.name, dtype=float)
    return pd.Series(np.asarray(list(y), dtype=float), dtype=float)


@dataclass
class TSResult:
    """Common StochX wrapper around a fitted statsmodels time-series result."""

    model_name: str
    fitted_model: Any
    result: Any
    original: TimeSeries | pd.Series
    order: tuple[int, int, int] | None = None
    seasonal_order: tuple[int, int, int, int] | None = None

    @property
    def params(self) -> pd.Series:
        """Return estimated coefficients."""
        return self.result.params

    @property
    def bse(self) -> pd.Series:
        """Return coefficient standard errors."""
        return self.result.bse

    @property
    def tvalues(self) -> pd.Series:
        """Return coefficient t-statistics."""
        return self.result.tvalues

    @property
    def pvalues(self) -> pd.Series:
        """Return coefficient p-values."""
        return self.result.pvalues

    @property
    def residuals(self) -> np.ndarray:
        """Return model residuals."""
        return np.asarray(self.result.resid, dtype=float)

    @property
    def fittedvalues(self) -> np.ndarray:
        """Return fitted values."""
        return np.asarray(self.result.fittedvalues, dtype=float)

    def forecast(self, steps: int = 1, alpha: float = 0.05) -> pd.DataFrame:
        """Forecast future values with prediction intervals."""
        if steps < 1:
            raise ValueError("steps must be positive")
        prediction = self.result.get_forecast(steps=steps)
        frame = prediction.summary_frame(alpha=alpha)
        return frame.rename(
            columns={"mean": "Forecast", "mean_ci_lower": "Lower", "mean_ci_upper": "Upper"}
        )

    def coefficients_table(self) -> pd.DataFrame:
        """Return an EViews-style coefficient table."""
        return pd.DataFrame(
            {
                "Coefficient": self.params,
                "Std. Error": self.bse,
                "t-Statistic": self.tvalues,
                "Prob.": self.pvalues,
            }
        )

    def statistics(self) -> dict[str, float]:
        """Return common EViews-style model statistics."""
        result = self.result
        values: dict[str, float] = {}
        for key, attr in {
            "R-squared": "rsquared",
            "Adjusted R-squared": "rsquared_adj",
            "S.E. of regression": "scale",
            "Sum squared resid": "ssr",
            "Log likelihood": "llf",
            "Akaike info criterion": "aic",
            "Schwarz criterion": "bic",
            "Hannan-Quinn criterion": "hqic",
        }.items():
            value = getattr(result, attr, np.nan)
            try:
                values[key] = float(value)
            except (TypeError, ValueError):
                values[key] = float("nan")
        return values

    def summary(self) -> str:
        """Render a compact EViews-like estimation report."""
        lines = [
            f"{self.model_name} Estimation Results",
            "=" * 72,
            f"Order: {self.order}" if self.order is not None else "",
        ]
        table = self.coefficients_table()
        lines.extend(["", table.to_string(float_format=lambda x: f"{x: .6f}")])
        lines.extend(["", "Model statistics"])
        for key, value in self.statistics().items():
            if np.isfinite(value):
                lines.append(f"{key:24s} {value: .6f}")
        return "\n".join(line for line in lines if line != "")

    def interpret(self, alpha: float = 0.05) -> str:
        """Interpret coefficient significance and model validity at ``alpha``."""
        statements = []
        significant = self.pvalues < alpha
        if significant.any():
            names = list(self.pvalues.index[significant])
            statements.append("Significant coefficients at the chosen level: " + ", ".join(map(str, names)) + ".")
        else:
            statements.append("No coefficient is statistically significant at the chosen level.")
        stats = self.statistics()
        if np.isfinite(stats.get("Akaike info criterion", np.nan)):
            statements.append("Compare AIC/BIC/HQ with competing models; lower values indicate the preferred specification under each criterion.")
        statements.append("Validate the residuals as a white-noise process before accepting the model for forecasting.")
        return " ".join(statements)

    def diagnostics(self, lags: int = 12):
        """Run the standard StochX residual validation battery."""
        from .diagnostics import residual_diagnostics

        return residual_diagnostics(self.residuals, lags=lags, p=self.order[0] if self.order else 0, q=self.order[2] if self.order else 0)

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


def fit_ar(y: TimeSeries | Iterable[float], p: int, *, trend: str = "c", method: str = "yule_walker") -> TSResult:
    """Estimate AR(p); the Yule-Walker route is the course's OLS-equivalent AR estimator."""
    if p < 1:
        raise ValueError("p must be positive")
    series = _as_series(y)
    from statsmodels.tsa.ar_model import AutoReg

    trend_map = {"n": "n", "c": "c", "ct": "ct"}
    if trend not in trend_map:
        raise ValueError("trend must be 'n', 'c', or 'ct'")
    model = AutoReg(series, lags=p, trend=trend_map[trend], old_names=False)
    result = model.fit()
    return TSResult("AR", model, result, y, order=(p, 0, 0))


def fit_ma(y: TimeSeries | Iterable[float], q: int, *, trend: str = "c") -> TSResult:
    """Estimate MA(q) under Gaussian maximum likelihood."""
    if q < 1:
        raise ValueError("q must be positive")
    series = _as_series(y)
    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(series, order=(0, 0, q), trend=trend)
    result = model.fit()
    return TSResult("MA", model, result, y, order=(0, 0, q))


def fit_arma(y: TimeSeries | Iterable[float], p: int, q: int, *, trend: str = "c") -> TSResult:
    """Estimate ARMA(p,q) by maximum likelihood."""
    if p < 0 or q < 0 or (p == 0 and q == 0):
        raise ValueError("at least one of p or q must be positive")
    series = _as_series(y)
    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(series, order=(p, 0, q), trend=trend)
    result = model.fit()
    return TSResult("ARMA", model, result, y, order=(p, 0, q))


def fit_arima(y: TimeSeries | Iterable[float], p: int, d: int, q: int, *, trend: str | None = None) -> TSResult:
    """Estimate ARIMA(p,d,q) with automatic handling of differencing."""
    if min(p, d, q) < 0:
        raise ValueError("p, d, and q must be non-negative")
    series = _as_series(y)
    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(series, order=(p, d, q), trend=trend)
    result = model.fit()
    return TSResult("ARIMA", model, result, y, order=(p, d, q))


def fit_sarima(
    y: TimeSeries | Iterable[float],
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    *,
    trend: str | None = None,
) -> TSResult:
    """Estimate SARIMA(p,d,q)(P,D,Q,s) with exact state-space likelihood."""
    series = _as_series(y)
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(series, order=order, seasonal_order=seasonal_order, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit(disp=False)
    return TSResult("SARIMA", model, result, y, order=order, seasonal_order=seasonal_order)


def estimate(
    y: TimeSeries | Iterable[float],
    *,
    p: int = 0,
    d: int = 0,
    q: int = 0,
    seasonal_order: tuple[int, int, int, int] | None = None,
) -> TSResult:
    """Unified EViews-like estimator dispatcher for AR/MA/ARMA/ARIMA/SARIMA."""
    if seasonal_order is not None:
        return fit_sarima(y, (p, d, q), seasonal_order)
    if d > 0:
        return fit_arima(y, p, d, q)
    if p and not q:
        return fit_ar(y, p)
    if q and not p:
        return fit_ma(y, q)
    return fit_arma(y, p, q)


def compare_orders(
    y: TimeSeries | Iterable[float],
    orders: Iterable[tuple[int, int, int]],
) -> pd.DataFrame:
    """Estimate competing ARIMA orders and return AIC/BIC/HQ ranking."""
    rows = []
    for order in orders:
        try:
            result = fit_arima(y, *order)
            stats = result.statistics()
            rows.append({"p": order[0], "d": order[1], "q": order[2], "AIC": stats["Akaike info criterion"], "BIC": stats["Schwarz criterion"], "HQ": stats["Hannan-Quinn criterion"], "LogLik": stats["Log likelihood"]})
        except Exception as exc:  # noqa: BLE001
            rows.append({"p": order[0], "d": order[1], "q": order[2], "AIC": np.nan, "BIC": np.nan, "HQ": np.nan, "LogLik": np.nan, "Error": str(exc)})
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["Rank by AIC"] = frame["AIC"].rank(method="min")
        frame["Rank by BIC"] = frame["BIC"].rank(method="min")
        frame["Rank by HQ"] = frame["HQ"].rank(method="min")
    return frame.sort_values("AIC", na_position="last") if not frame.empty else frame
