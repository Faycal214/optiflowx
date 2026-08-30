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

    def _eviews_parameter(self, attribute: str) -> pd.Series:
        raw = getattr(super(), attribute)
        if not isinstance(raw, pd.Series):
            raw = pd.Series(raw)
        mapping = {}
        for name in raw.index:
            text = str(name)
            if text.startswith("ar.L"):
                mapping[name] = f"AR({int(text[4:])})"
            elif text.startswith("ma.L"):
                mapping[name] = f"MA({int(text[4:])})"
            elif text.startswith("ar.S.L"):
                mapping[name] = f"SAR({int(text[6:])})"
            elif text.startswith("ma.S.L"):
                mapping[name] = f"SMA({int(text[6:])})"
            elif text.lower() in {"sigma2", "sigmasq"}:
                mapping[name] = "SIGMASQ"
        return raw.rename(index=mapping)

    @property
    def params(self) -> pd.Series:
        return self._eviews_parameter("params")

    @property
    def bse(self) -> pd.Series:
        return self._eviews_parameter("bse").reindex(self.params.index)

    @property
    def tvalues(self) -> pd.Series:
        return self._eviews_parameter("tvalues").reindex(self.params.index)

    @property
    def pvalues(self) -> pd.Series:
        return self._eviews_parameter("pvalues").reindex(self.params.index)

    @property
    def params_eviews(self) -> pd.Series:
        return self.params
    @property
    def model_name(self) -> str:
        """Return the model family name."""
        return self.title

    def forecast(self, steps: int = 1, alpha: float = 0.05) -> pd.DataFrame:
        """Forecast future values with prediction intervals.

        ARIMA/SARIMAX results expose ``get_forecast`` while AutoReg exposes
        ``get_prediction``. Both paths are normalized to the same result frame.
        """
        if steps < 1:
            raise ValueError("steps must be positive")

        if hasattr(self.result, "get_forecast"):
            prediction = self.result.get_forecast(steps=steps)
        elif hasattr(self.result, "get_prediction"):
            nobs = int(
                getattr(
                    self.result,
                    "nobs",
                    len(self.original) if self.original is not None else 0,
                )
            )
            prediction = self.result.get_prediction(
                start=nobs,
                end=nobs + steps - 1,
                dynamic=False,
            )
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
        """Return EViews-style inverted AR and MA roots."""
        ar_roots = np.asarray(getattr(self.result, "arroots", []), dtype=complex)
        ma_roots = np.asarray(getattr(self.result, "maroots", []), dtype=complex)
        return {
            "AR roots": 1.0 / ar_roots if ar_roots.size else ar_roots,
            "MA roots": 1.0 / ma_roots if ma_roots.size else ma_roots,
        }

    def stability(self) -> dict[str, bool]:
        """Check EViews inverse-root stationarity and invertibility conditions."""
        roots = self.roots()
        ar_ok = bool(np.all(np.abs(roots["AR roots"]) < 1.0)) if roots["AR roots"].size else True
        ma_ok = bool(np.all(np.abs(roots["MA roots"]) < 1.0)) if roots["MA roots"].size else True
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


def _result(model_name: str, model: Any, result: Any, y, order, seasonal_order=None, method: str = "Maximum Likelihood") -> TSResult:
    return TSResult(
        result=result,
        title=model_name,
        dependent=getattr(y, "name", "Y"),
        method=method,
        fitted_model=model,
        original=y,
        order=order,
        seasonal_order=seasonal_order,
    )


def fit_ar(y: TimeSeries | Iterable[float], p: int | Iterable[int], *, trend: str = "c", method: str = "ml", optimizer: str = "bfgs", covariance: str = "opg", maxiter: int = 1000) -> TSResult:
    """Estimate AR(p) using the EViews default ML method."""
    if p < 1:
        raise ValueError("p must be positive")
    series = _as_series(y)
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    if method.lower() != "ml":
        raise ValueError("fit_ar currently supports method='ml'; CLS/GLS are not yet implemented")
    if trend not in {"n", "c", "ct"}:
        raise ValueError("trend must be 'n', 'c', or 'ct'")
    if optimizer.lower() != "bfgs" or covariance.lower() != "opg":
        raise ValueError("only optimizer='bfgs' and covariance='opg' are currently implemented")
    if not isinstance(maxiter, int) or isinstance(maxiter, bool) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    model = SARIMAX(series, order=(p, 0, 0), trend=trend, enforce_stationarity=True, enforce_invertibility=True)
    result = model.fit(method=optimizer.lower(), maxiter=maxiter, disp=False, cov_type=covariance.lower())
    return _result("AR", model, result, y, (p, 0, 0))
def fit_ma(y: TimeSeries | Iterable[float], q: int | Iterable[int], *, trend: str = "c", method: str = "ml", optimizer: str = "bfgs", covariance: str = "opg", maxiter: int = 1000) -> TSResult:
    """Estimate MA(q) using the EViews default ML method."""
    if q < 1:
        raise ValueError("q must be positive")
    series = _as_series(y)
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    if method.lower() != "ml":
        raise ValueError("fit_ma currently supports method='ml'; CLS/GLS are not yet implemented")
    if optimizer.lower() != "bfgs" or covariance.lower() != "opg":
        raise ValueError("only optimizer='bfgs' and covariance='opg' are currently implemented")
    if not isinstance(maxiter, int) or isinstance(maxiter, bool) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    if trend not in {"n", "c", "ct"}:
        raise ValueError("trend must be 'n', 'c', or 'ct'")
    model = SARIMAX(series, order=(0, 0, q), trend=trend, enforce_stationarity=True, enforce_invertibility=True)
    result = model.fit(method=optimizer.lower(), maxiter=maxiter, disp=False, cov_type=covariance.lower())
    return _result("MA", model, result, y, (0, 0, q))
def fit_arma(y: TimeSeries | Iterable[float], p: int | Iterable[int], q: int | Iterable[int], *, trend: str = "c", method: str = "ml", optimizer: str = "bfgs", covariance: str = "opg", maxiter: int = 1000) -> TSResult:
    """Estimate ARMA(p,q) using the EViews default ML method."""
    if p < 0 or q < 0 or (p == 0 and q == 0):
        raise ValueError("at least one of p or q must be positive")
    series = _as_series(y)
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    if method.lower() != "ml":
        raise ValueError("fit_arma currently supports method='ml'; CLS/GLS are not yet implemented")
    if optimizer.lower() != "bfgs" or covariance.lower() != "opg":
        raise ValueError("only optimizer='bfgs' and covariance='opg' are currently implemented")
    if not isinstance(maxiter, int) or isinstance(maxiter, bool) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    if trend not in {"n", "c", "ct"}:
        raise ValueError("trend must be 'n', 'c', or 'ct'")
    model = SARIMAX(series, order=(p, 0, q), trend=trend, enforce_stationarity=True, enforce_invertibility=True)
    result = model.fit(method="bfgs", maxiter=1000, disp=False)
    return _result("ARMA", model, result, y, (p, 0, q))
def fit_arima(
    y: TimeSeries | Iterable[float],
    p: int, d: int, q: int,
    *, trend: str | None = "c", method: str = "ml",
    optimizer: str = "bfgs", covariance: str = "opg", maxiter: int = 1000,
) -> TSResult:
    """Estimate EViews-style ARIMA(p,d,q) by state-space maximum likelihood."""
    if min(p, d, q) < 0:
        raise ValueError("p, d, and q must be non-negative")
    if p == 0 and q == 0 and d == 0:
        raise ValueError("ARIMA(0,0,0) is a mean specification; use OLS or a constant series model")
    if method.lower() != "ml":
        raise ValueError("only EViews ML is currently implemented for direct ARIMA")
    if optimizer.lower() != "bfgs" or covariance.lower() != "opg":
        raise ValueError("only EViews-default BFGS optimizer and OPG covariance are implemented")
    if trend not in {None, "n", "c", "ct"}:
        raise ValueError("trend must be None, n, c, or ct")
    series = _as_series(y)
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    model = SARIMAX(
        series, order=(p, d, q), trend=trend,
        enforce_stationarity=True, enforce_invertibility=True,
    )
    result = model.fit(method="bfgs", maxiter=maxiter, disp=False, cov_type="opg")
    return _result("ARIMA", model, result, y, (p, d, q), method="Maximum Likelihood")

def fit_sarima(
    y: TimeSeries | Iterable[float],
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    *, trend: str | None = "c", method: str = "ml",
    optimizer: str = "bfgs", covariance: str = "opg", maxiter: int = 1000,
) -> TSResult:
    """Estimate multiplicative SARIMA with EViews-style ML defaults."""
    p, d, q = order
    P, D, Q, s = seasonal_order
    if min(p, d, q, P, D, Q) < 0 or s < 1:
        raise ValueError("SARIMA orders must be non-negative and seasonal period must be positive")
    if method.lower() != "ml":
        raise ValueError("only EViews ML is implemented for direct SARIMA")
    if optimizer.lower() != "bfgs" or covariance.lower() != "opg":
        raise ValueError("only EViews-default BFGS/OPG is implemented for direct SARIMA")
    if trend not in {None, "n", "c", "ct"}:
        raise ValueError("trend must be None, n, c, or ct")
    series = _as_series(y)
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    model = SARIMAX(
        series, order=order, seasonal_order=seasonal_order, trend=trend,
        enforce_stationarity=True, enforce_invertibility=True,
    )
    result = model.fit(method="bfgs", maxiter=maxiter, disp=False, cov_type="opg")
    return _result("SARIMA", model, result, y, order, seasonal_order, method="Maximum Likelihood")


def estimate(
    y: TimeSeries | Iterable[float],
    *,
    p: int = 0,
    d: int = 0,
    q: int = 0,
    seasonal_order: tuple[int, int, int, int] | None = None,
) -> TSResult:
    """Unified estimator dispatcher for AR/MA/ARMA/ARIMA/SARIMA."""
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
            rows.append({
                "p": order[0],
                "d": order[1],
                "q": order[2],
                "AIC": stats["Akaike info criterion"],
                "BIC": stats["Schwarz criterion"],
                "HQ": stats["Hannan-Quinn criterion"],
                "LogLik": stats["Log likelihood"],
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "p": order[0], "d": order[1], "q": order[2],
                "AIC": np.nan, "BIC": np.nan, "HQ": np.nan,
                "LogLik": np.nan, "Error": str(exc),
            })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["Rank by AIC"] = frame["AIC"].rank(method="min")
        frame["Rank by BIC"] = frame["BIC"].rank(method="min")
        frame["Rank by HQ"] = frame["HQ"].rank(method="min")
    return frame.sort_values("AIC", na_position="last") if not frame.empty else framedef _lag_tuple(value: int | Iterable[int], name: str, *, allow_empty: bool = False) -> tuple[int, ...]:
    """Normalize an EViews lag specification while preserving sparse lags."""
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0 or (value == 0 and not allow_empty):
            raise ValueError(f"{name} must be positive")
        return tuple(range(1, value + 1))
    lags = tuple(sorted(set(int(v) for v in value)))
    if (not lags and not allow_empty) or any(v < 1 for v in lags):
        raise ValueError(f"{name} must contain positive lag orders")
    return lags

def fit_ar(y: TimeSeries | Iterable[float], p: int | Iterable[int], *, trend: str = "c", method: str = "ml", optimizer: str = "bfgs", covariance: str = "opg", maxiter: int = 1000) -> TSResult:
    """Estimate AR terms using the EViews ML/BFGS/OPG defaults."""
    p_lags = _lag_tuple(p, "p")
    series = _as_series(y)
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    if method.lower() != "ml":
        raise ValueError("only EViews ARMA method='ml' is implemented")
    if optimizer.lower() != "bfgs" or covariance.lower() != "opg":
        raise ValueError("only EViews-default optimizer='bfgs' and covariance='opg' are implemented")
    if not isinstance(maxiter, int) or isinstance(maxiter, bool) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    if trend not in {"n", "c", "ct"}:
        raise ValueError("trend must be 'n', 'c', or 'ct'")
    model = SARIMAX(series, order=(p_lags, 0, ()), trend=trend, enforce_stationarity=True, enforce_invertibility=True)
    result = model.fit(method="bfgs", maxiter=maxiter, disp=False, cov_type="opg")
    return _result("AR", model, result, y, (max(p_lags), 0, 0))

def fit_ma(y: TimeSeries | Iterable[float], q: int | Iterable[int], *, trend: str = "c", method: str = "ml", optimizer: str = "bfgs", covariance: str = "opg", maxiter: int = 1000) -> TSResult:
    """Estimate MA terms using the EViews ML/BFGS/OPG defaults."""
    q_lags = _lag_tuple(q, "q")
    series = _as_series(y)
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    if method.lower() != "ml":
        raise ValueError("only EViews ARMA method='ml' is implemented")
    if optimizer.lower() != "bfgs" or covariance.lower() != "opg":
        raise ValueError("only EViews-default optimizer='bfgs' and covariance='opg' are implemented")
    if not isinstance(maxiter, int) or isinstance(maxiter, bool) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    if trend not in {"n", "c", "ct"}:
        raise ValueError("trend must be 'n', 'c', or 'ct'")
    model = SARIMAX(series, order=((), 0, q_lags), trend=trend, enforce_stationarity=True, enforce_invertibility=True)
    result = model.fit(method="bfgs", maxiter=maxiter, disp=False, cov_type="opg")
    return _result("MA", model, result, y, (0, 0, max(q_lags)))

def fit_arma(y: TimeSeries | Iterable[float], p: int | Iterable[int], q: int | Iterable[int], *, trend: str = "c", method: str = "ml", optimizer: str = "bfgs", covariance: str = "opg", maxiter: int = 1000) -> TSResult:
    """Estimate sparse/dense ARMA terms using EViews ML/BFGS/OPG defaults."""
    p_lags = _lag_tuple(p, "p", allow_empty=isinstance(p, int) and p == 0)
    q_lags = _lag_tuple(q, "q", allow_empty=isinstance(q, int) and q == 0)
    if not p_lags and not q_lags:
        raise ValueError("at least one AR or MA lag must be present")
    series = _as_series(y)
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    if method.lower() != "ml":
        raise ValueError("only EViews ARMA method='ml' is implemented")
    if optimizer.lower() != "bfgs" or covariance.lower() != "opg":
        raise ValueError("only EViews-default optimizer='bfgs' and covariance='opg' are implemented")
    if not isinstance(maxiter, int) or isinstance(maxiter, bool) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    if trend not in {"n", "c", "ct"}:
        raise ValueError("trend must be 'n', 'c', or 'ct'")
    model = SARIMAX(series, order=(p_lags, 0, q_lags), trend=trend, enforce_stationarity=True, enforce_invertibility=True)
    result = model.fit(method="bfgs", maxiter=maxiter, disp=False, cov_type="opg")
    return _result("ARMA", model, result, y, (max(p_lags, default=0), 0, max(q_lags, default=0)))
def fit_arima(y: TimeSeries | Iterable[float], p: int, d: int, q: int, *, trend: str | None = None) -> TSResult:
    """Estimate ARIMA(p,d,q) with automatic handling of differencing."""
    if min(p, d, q) < 0:
        raise ValueError("p, d, and q must be non-negative")
    series = _as_series(y)
    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(series, order=(p, d, q), trend=trend)
    result = model.fit()
    return _result("ARIMA", model, result, y, (p, d, q))


def fit_sarima(
    y: TimeSeries | Iterable[float],
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    *,
    trend: str | None = None,
) -> TSResult:
    """Estimate SARIMA(p,d,q)(P,D,Q,s) with state-space likelihood."""
    series = _as_series(y)
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)
    return _result("SARIMA", model, result, y, order, seasonal_order)


def estimate(
    y: TimeSeries | Iterable[float],
    *,
    p: int = 0,
    d: int = 0,
    q: int = 0,
    seasonal_order: tuple[int, int, int, int] | None = None,
) -> TSResult:
    """Unified estimator dispatcher for AR/MA/ARMA/ARIMA/SARIMA."""
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
            rows.append({
                "p": order[0],
                "d": order[1],
                "q": order[2],
                "AIC": stats["Akaike info criterion"],
                "BIC": stats["Schwarz criterion"],
                "HQ": stats["Hannan-Quinn criterion"],
                "LogLik": stats["Log likelihood"],
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "p": order[0], "d": order[1], "q": order[2],
                "AIC": np.nan, "BIC": np.nan, "HQ": np.nan,
                "LogLik": np.nan, "Error": str(exc),
            })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["Rank by AIC"] = frame["AIC"].rank(method="min")
        frame["Rank by BIC"] = frame["BIC"].rank(method="min")
        frame["Rank by HQ"] = frame["HQ"].rank(method="min")
    return frame.sort_values("AIC", na_position="last") if not frame.empty else frame
