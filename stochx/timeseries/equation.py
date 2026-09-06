"""EViews-inspired equation specifications and estimation objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .arma_errors import ErrorProcess, parse_error_terms
from .expression import ExpressionError, evaluate
from .results import UnifiedResult
from .arma_estimation import fit_cls, fit_gls, make_starting_values

_RANGE_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\(\s*(?P<start>-?\d+)\s+to\s+(?P<end>-?\d+)\s*\)$", re.IGNORECASE)


def _expand_eviews_ranges(specification: str) -> list[str]:
    """Expand EViews lag/lead ranges such as ``CPI(0 to -12)``."""
    tokens = specification.split()
    expanded: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if i + 2 < len(tokens) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\(\s*-?\d+$", token) and tokens[i + 1].lower() == "to":
            candidate = f"{token} {tokens[i + 1]} {tokens[i + 2]}".replace(" ", "")
            match = re.match(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((-?\d+)to(-?\d+)\)$", candidate, re.IGNORECASE)
            if match:
                name = match.group("name")
                start = int(match.group(2))
                end = int(match.group(3))
                step = 1 if end >= start else -1
                for offset in range(start, end + step, step):
                    expanded.append(name if offset == 0 else f"{name}({offset})")
                i += 3
                continue
        match = _RANGE_RE.match(token)
        if match:
            name = match.group("name")
            start = int(match.group("start"))
            end = int(match.group("end"))
            step = 1 if end >= start else -1
            for offset in range(start, end + step, step):
                expanded.append(name if offset == 0 else f"{name}({offset})")
        else:
            expanded.append(token)
        i += 1
    return expanded


def _rename_arma_parameters(series: pd.Series, error_process: ErrorProcess) -> pd.Series:
    """Map statsmodels ARMA parameter labels onto EViews names."""
    if not (error_process.max_p or error_process.max_q):
        return series
    mapping: dict[str, str] = {}
    for name in series.index:
        text = str(name)
        if text.startswith("ar.L"):
            mapping[text] = f"AR({int(text[4:])})"
        elif text.startswith("ma.L"):
            mapping[text] = f"MA({int(text[4:])})"
        elif text.startswith("ar.S.L"):
            mapping[text] = f"SAR({int(text[6:])})"
        elif text.startswith("ma.S.L"):
            mapping[text] = f"SMA({int(text[6:])})"
        elif text.lower() in {"sigma2", "sigmasq"}:
            mapping[text] = "SIGMASQ"
    return series.rename(index=mapping)


@dataclass
class EquationResult(UnifiedResult):
    """OLS or ARMA-error equation result with EViews-style output."""

    specification: str = ""
    error_process: ErrorProcess = ErrorProcess()
    _opg_covariance: pd.DataFrame | None = None
    observed: np.ndarray | None = None
    workfile: Any = None

    def _parameter_series(self, attribute: str) -> pd.Series:
        if attribute == "bse" and self._opg_covariance is not None:
            return pd.Series(np.sqrt(np.clip(np.diag(self._opg_covariance.to_numpy(dtype=float)), 0.0, None)), index=self._opg_covariance.index)
        if attribute in {"tvalues", "pvalues"} and self._opg_covariance is not None:
            params = self.params
            bse = self.bse
            tvalues = params / bse
            if attribute == "tvalues":
                return tvalues
            return pd.Series(2.0 * scipy_stats.norm.sf(np.abs(tvalues.to_numpy(dtype=float))), index=tvalues.index)
        value = getattr(self.result, attribute, pd.Series(dtype=float))
        if isinstance(value, pd.Series):
            series = value.astype(float)
        else:
            array = np.asarray(value, dtype=float).reshape(-1)
            names = getattr(self.result, "param_names", None)
            if names is None:
                names = getattr(getattr(self.result, "model", None), "param_names", None)
            if names is None or len(names) != len(array):
                names = list(range(len(array)))
            series = pd.Series(array, index=names, dtype=float)
        return _rename_arma_parameters(series, self.error_process)

    @property
    def params(self) -> pd.Series:
        return self._parameter_series("params")

    @property
    def bse(self) -> pd.Series:
        return self._parameter_series("bse").reindex(self.params.index)

    @property
    def tvalues(self) -> pd.Series:
        return self._parameter_series("tvalues").reindex(self.params.index)

    @property
    def pvalues(self) -> pd.Series:
        return self._parameter_series("pvalues").reindex(self.params.index)

    @property
    def convergence(self) -> dict[str, Any]:
        """Expose optimizer convergence metadata in an EViews-like form."""
        retvals = getattr(self.result, "mle_retvals", {}) or {}
        converged = retvals.get("converged")
        iterations = retvals.get("iterations", retvals.get("nit"))
        return {
            "converged": None if converged is None else bool(converged),
            "iterations": None if iterations is None else int(iterations),
            "optimizer": "BFGS",
            "information_matrix": "OPG",
            "degree_of_freedom_adjustment": False,
        }

    def summary(self) -> str:
        """Render the main EViews equation estimation table."""
        from .reporting import render_eviews
        return render_eviews(self)

    def coefficient_output(self) -> pd.DataFrame:
        """EViews coefficient table as a DataFrame."""
        from .reporting import EViewsReport
        return EViewsReport(self).coefficient_table()

    def report(self, *, include_covariance: bool = False, include_diagnostics: bool = False, diagnostic_lags: int = 12) -> str:
        """Render a combined EViews-style equation report."""
        from .reporting import EViewsReport
        renderer = EViewsReport(self)
        sections = [renderer.text()]
        if include_covariance:
            sections.append("Coefficient Covariance Matrix\n" + self.covariance.to_string(float_format=lambda x: f"{x:.6f}"))
        if include_diagnostics:
            sections.append(renderer.diagnostic_text("Residual Diagnostics", self.diagnostics(lags=diagnostic_lags)))
        return "\n\n".join(sections)
    @property
    def covariance_method(self) -> str:
        if self._opg_covariance is not None:
            return "outer product of gradients (OPG)"
        if hasattr(self.result, "covariance"):
            return "ordinary"
        return "model default"

    def forecast(
        self, *, steps: int | None = None, start: int | None = None, end: int | None = None,
        dynamic: bool = False, structural: bool = False, alpha: float = 0.05,
        coef_uncertainty: bool = True, ma_backcast: str = "estimation",
        forecast_fill: str = "actual", future_exog=None, actuals=None,
    ) -> pd.DataFrame:
        """Forecast this equation using EViews-style controls."""
        if not 0 < float(alpha) < 1: raise ValueError("alpha must lie strictly between 0 and 1")
        if ma_backcast.lower() not in {"estimation", "forecast_available", "fa"}: raise ValueError("invalid ma_backcast")
        if forecast_fill.lower() not in {"actual", "na"}: raise ValueError("forecast_fill must be actual or na")
        model = self.result
        nobs = int(getattr(model, "nobs", len(self.observed) if self.observed is not None else 0))
        if steps is not None and (start is not None or end is not None): raise ValueError("use steps or start/end")
        if steps is not None:
            if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1: raise ValueError("steps must be a positive integer")
            start_i, end_i = nobs, nobs + steps - 1
        else:
            if start is None: raise ValueError("steps or start is required")
            start_i, end_i = int(start), int(end if end is not None else start)
            if end_i < start_i: raise ValueError("end must be >= start")
        horizon = end_i - start_i + 1
        if structural:
            model_exog = np.asarray(getattr(getattr(model, "model", None), "exog", np.empty((0,0))), dtype=float)
            arma_labels = {f"AR({i})" for i in self.error_process.p} | {f"MA({i})" for i in self.error_process.q} | {f"SAR({i})" for i in self.error_process.sar} | {f"SMA({i})" for i in self.error_process.sma} | {"SIGMASQ"}
            beta_names = [name for name in self.params.index if str(name) not in arma_labels]
            beta = np.asarray([self.params[name] for name in beta_names], dtype=float)
            if future_exog is not None:
                Xf = np.asarray(future_exog, dtype=float); Xf = Xf.reshape(1,-1) if Xf.ndim == 1 else Xf
            elif start_i < nobs:
                Xf = model_exog[start_i:end_i+1]
            elif beta_names == ["C"]:
                Xf = np.ones((horizon, 1))
            else:
                raise ValueError("future_exog is required for out-of-sample structural forecasts")
            if Xf.shape[1] != beta.size: raise ValueError("future_exog columns do not match structural coefficients")
            point = Xf @ beta
            se = np.full(horizon, np.nan); lower = np.full(horizon, np.nan); upper = np.full(horizon, np.nan)
        else:
            model_exog = np.asarray(getattr(getattr(model, "model", None), "exog", np.empty((0, 0))), dtype=float)
            if not self.error_process.p and not self.error_process.q and not self.error_process.sar and not self.error_process.sma:
                if future_exog is not None:
                    exog = np.asarray(future_exog, dtype=float)
                    exog = exog.reshape(1, -1) if exog.ndim == 1 else exog
                elif start_i < nobs:
                    exog = model_exog[start_i:end_i + 1]
                else:
                    beta_names = list(self.params.index)
                    if beta_names == ["C"]:
                        exog = np.ones((horizon, 1))
                    else:
                        raise ValueError("future_exog is required for out-of-sample forecasts")
                if exog.shape[0] != horizon:
                    raise ValueError("forecast exog length must equal forecast horizon")
                prediction = model.get_prediction(exog=exog)
            else:
                kwargs = {}
                if future_exog is not None: kwargs["exog"] = np.asarray(future_exog, dtype=float)
                prediction = model.get_prediction(start=start_i, end=end_i, dynamic=dynamic, **kwargs)
            frame = prediction.summary_frame(alpha=alpha)
            point = frame["mean"].to_numpy(dtype=float)
            if "obs_ci_lower" in frame and "obs_ci_upper" in frame:
                lower = frame["obs_ci_lower"].to_numpy(dtype=float)
                upper = frame["obs_ci_upper"].to_numpy(dtype=float)
                z = float(scipy_stats.norm.ppf(1.0 - alpha / 2.0))
                se = (upper - lower) / (2.0 * z)
            else:
                lower = frame["mean_ci_lower"].to_numpy(dtype=float)
                upper = frame["mean_ci_upper"].to_numpy(dtype=float)
                se = frame["mean_se"].to_numpy(dtype=float) if "mean_se" in frame else np.full(horizon, np.nan)
        out = pd.DataFrame({"Forecast": point, "Std. Error": se, "Lower": lower, "Upper": upper})
        if actuals is not None:
            actual = np.asarray(actuals, dtype=float).reshape(-1)
            if actual.size != horizon: raise ValueError("actuals length must equal forecast horizon")
            out["Actual"] = actual; out["Error"] = actual - out["Forecast"]
        out.attrs.update({"forecast_sample": (start_i, end_i), "dynamic": dynamic, "structural": structural, "coef_uncertainty": coef_uncertainty, "ma_backcast": ma_backcast, "forecast_fill": forecast_fill})
        return out

    def fit(self, *, start: int | None = None, end: int | None = None, alpha: float = 0.05, structural: bool = False, coef_uncertainty: bool = True, future_exog=None, actuals=None) -> pd.DataFrame:
        """EViews static fitted/forecast procedure (actual lagged values)."""
        if start is None and end is None:
            nobs = int(getattr(self.result, "nobs", len(self.observed) if self.observed is not None else 0))
            start = 0
            end = nobs - 1
        return self.forecast(start=start, end=end, dynamic=False, structural=structural, alpha=alpha, coef_uncertainty=coef_uncertainty, future_exog=future_exog, actuals=actuals)

    def view_text(self, view: str = "estimate") -> str:
        """Render a named EViews-style equation view as text."""
        key = view.lower().replace(" ", "_").replace("-", "_")
        from .reporting import EViewsReport
        renderer = EViewsReport(self)
        if key in {"estimate", "output", "summary"}:
            return renderer.text()
        if key in {"covariance", "covariance_matrix"}:
            return "Coefficient Covariance Matrix\n" + self.covariance.to_string(float_format=lambda x: f"{x:.6f}")
        if key in {"residual_correlogram", "correlogram_q_statistics"}:
            return renderer.diagnostic_text("Correlogram - Q-statistics", self.residual_correlogram())
        if key in {"squared_residual_correlogram", "correlogram_squared_residuals"}:
            return renderer.diagnostic_text("Correlogram of Squared Residuals", self.squared_residual_correlogram())
        if key in {"histogram_normality", "histogram_normality_test"}:
            return renderer.diagnostic_text("Histogram-Normality", self.normality_test())
        if key in {"serial_correlation_lm", "auto"}:
            return renderer.diagnostic_text("Breusch-Godfrey Serial Correlation LM Test", self.serial_correlation_lm(1))
        if key in {"heteroskedasticity", "hettest"}:
            return renderer.diagnostic_text("Heteroskedasticity Test", self.heteroskedasticity())
        raise ValueError(f"unknown equation view: {view}")
    def forecast_evaluation(self, forecast, actual) -> dict[str, float]:
        """Evaluate forecasts with the EViews four-statistic report."""
        f = np.asarray(list(forecast), dtype=float)
        y = np.asarray(list(actual), dtype=float)
        if f.shape != y.shape:
            raise ValueError("forecast and actual must have the same shape")
        mask = np.isfinite(f) & np.isfinite(y)
        f, y = f[mask], y[mask]
        if f.size == 0:
            raise ValueError("no finite forecast/actual observations")
        e = y - f
        mse = float(np.mean(e ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(e)))
        nonzero = np.abs(y) > np.finfo(float).eps
        mape = float(np.mean(np.abs(e[nonzero] / y[nonzero])) * 100.0) if np.any(nonzero) else float("nan")
        theil = rmse / (float(np.sqrt(np.mean(y ** 2))) + float(np.sqrt(np.mean(f ** 2))))
        y_mean, f_mean = float(np.mean(y)), float(np.mean(f))
        y_sd = float(np.std(y, ddof=0))
        f_sd = float(np.std(f, ddof=0))
        corr = float(np.corrcoef(y, f)[0, 1]) if y.size > 1 and y_sd > 0 and f_sd > 0 else 0.0
        denom = mse if mse > np.finfo(float).eps else np.nan
        bias_prop = float((f_mean - y_mean) ** 2 / denom) if np.isfinite(denom) else 0.0
        variance_prop = float((f_sd - y_sd) ** 2 / denom) if np.isfinite(denom) else 0.0
        covariance_prop = float(1.0 - bias_prop - variance_prop) if np.isfinite(denom) else 0.0
        return {
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "Theil Inequality Coefficient": theil,
            "Bias Proportion": bias_prop,
            "Variance Proportion": variance_prop,
            "Covariance Proportion": covariance_prop,
            "Mean Error": float(np.mean(e)),
            "Observations": float(f.size),
        }

    def coint(self, *, method: str = "eg", x=None, trend: str = "const", lag=None, maxlag=None):
        """Run an EViews equation-level residual cointegration test."""
        from .cointegration import engle_granger, phillips_ouliaris
        if x is None: raise ValueError("x must be supplied")
        if method.lower() == "eg": return engle_granger(self.observed, x, trend=trend, lag=lag, maxlag=maxlag)
        if method.lower() == "po": return phillips_ouliaris(self.observed, x, trend=trend, lag=lag, maxlag=maxlag)
        raise ValueError("method must be 'eg' or 'po'")

    def cointreg(self, x, *, method: str = "fmols", trend: str = "const", leads: int = 0, lags: int = 0, kernel: str = "bartlett", bandwidth=None):
        """Estimate an EViews single-equation cointegrating regression."""
        from .cointegration import cointreg
        return cointreg(self.observed, x, method=method, trend=trend, leads=leads, lags=lags, kernel=kernel, bandwidth=bandwidth)

    def ecm(self, x, *, lags: int = 1, trend: str = "const"):
        """Estimate a single-equation error-correction model."""
        from .cointegration import ecm
        return ecm(self.observed, x, lags=lags, trend=trend)
    def diagnostics(self, *, lags: int = 12, alpha: float = 0.05, het_test: str = "BPG", white_cross_terms: bool = False) -> dict[str, object]:
        """Return EViews-style residual diagnostics for this equation."""
        from .diagnostics import histogram_normality, residual_correlogram, residual_correlogram_squared, serial_correlation_lm, heteroskedasticity_test
        out = {}
        model_df = len(self.error_process.p) + len(self.error_process.q) + len(self.error_process.sar) + len(self.error_process.sma)
        out["Correlogram-Q statistics"] = residual_correlogram(self.residuals, lags=lags, model_df=model_df, alpha=alpha)
        out["Squared residual correlogram"] = residual_correlogram_squared(self.residuals, lags=lags, model_df=model_df, alpha=alpha)
        out["Histogram-Normality"] = histogram_normality(self.residuals, alpha=alpha)
        if hasattr(self.result, "model") and getattr(self.result.model, "exog", None) is not None:
            X = np.asarray(self.result.model.exog, dtype=float)
            out["Heteroskedasticity"] = heteroskedasticity_test(self.residuals, X, test=het_test, lags=lags, cross_terms=white_cross_terms, alpha=alpha)
            try:
                out["Serial Correlation LM"] = serial_correlation_lm(self.residuals, X, lags=lags, alpha=alpha, model_df=model_df)
            except ValueError:
                pass
        return out

    def residual_correlogram(self, *, lags: int = 12, alpha: float = 0.05):
        from .diagnostics import residual_correlogram
        model_df = len(self.error_process.p) + len(self.error_process.q) + len(self.error_process.sar) + len(self.error_process.sma)
        return residual_correlogram(self.residuals, lags=lags, model_df=model_df, alpha=alpha)
