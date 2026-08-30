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

    def squared_residual_correlogram(self, *, lags: int = 12, alpha: float = 0.05):
        from .diagnostics import residual_correlogram_squared
        return residual_correlogram_squared(self.residuals, lags=lags, model_df=0, alpha=alpha)

    def serial_correlation_lm(self, lags: int = 1, *, alpha: float = 0.05):
        from .diagnostics import serial_correlation_lm
        model_exog = getattr(self.result.model, "exog", None)
        if model_exog is None:
            raise ValueError("serial correlation LM requires regression regressors")
        return serial_correlation_lm(self.residuals, model_exog, lags=lags, alpha=alpha, model_df=len(self.error_process.p)+len(self.error_process.q))

    def heteroskedasticity(self, *, test: str = "BPG", lags: int = 12, cross_terms: bool = False, alpha: float = 0.05):
        from .diagnostics import heteroskedasticity_test
        model_exog = getattr(self.result.model, "exog", None)
        if model_exog is None and test.upper() != "ARCH":
            raise ValueError("this heteroskedasticity test requires equation regressors")
        return heteroskedasticity_test(self.residuals, model_exog if model_exog is not None else np.ones((len(self.residuals), 1)), test=test, lags=lags, cross_terms=cross_terms, alpha=alpha)

    def normality_test(self, *, alpha: float = 0.05):
        from .diagnostics import histogram_normality
        return histogram_normality(self.residuals, alpha=alpha)

    def chow_breakpoint(self, breakpoint: int, *, alpha: float = 0.05):
        if self.error_process.max_p or self.error_process.max_q or self.error_process.max_sar or self.error_process.max_sma:
            raise ValueError("EViews Chow breakpoint stability view is restricted here to ordinary LS equations")
        from .diagnostics import chow_breakpoint
        y = np.asarray(self.result.model.endog, dtype=float).reshape(-1)
        X = np.asarray(self.result.model.exog, dtype=float)
        return chow_breakpoint(y, X, breakpoint, alpha=alpha)
    def stability_diagnostics(self, *, breakpoint: int | None = None, forecast_start: int | None = None, alpha: float = 0.05) -> dict[str, object]:
        """Return EViews-style stability diagnostics where supported."""
        if self.error_process.max_p or self.error_process.max_q or self.error_process.max_sar or self.error_process.max_sma:
            raise ValueError("recursive OLS stability diagnostics are restricted to equations without ARMA terms")
        from .diagnostics import chow_breakpoint, chow_forecast, cusum_tests, recursive_coefficient_estimates
        y = np.asarray(self.result.model.endog, dtype=float).reshape(-1)
        X = np.asarray(self.result.model.exog, dtype=float)
        out = {"Recursive Estimates": cusum_tests(y, X, alpha=alpha)}
out["Recursive Estimates"]["coefficients"] = recursive_coefficient_estimates(y, X)
        if breakpoint is not None:
            out["Chow Breakpoint"] = chow_breakpoint(y, X, breakpoint, alpha=alpha)
        if forecast_start is not None:
            out["Chow Forecast"] = chow_forecast(y, X, forecast_start, alpha=alpha)
        return out
    def covariance_matrix(self) -> pd.DataFrame:
        if self._opg_covariance is not None:
            return self._opg_covariance.copy()
        covariance = getattr(self.result, "cov_params", None)
        if covariance is None and hasattr(self.result, "covariance"):
            value = self.result.covariance
            return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(np.asarray(value, dtype=float), index=self.params.index, columns=self.params.index)
        if callable(covariance):
            values = np.asarray(covariance(), dtype=float)
            return pd.DataFrame(values, index=self.params.index, columns=self.params.index)
        return pd.DataFrame(np.asarray(covariance, dtype=float), index=self.params.index, columns=self.params.index)

    @property
    def covariance(self) -> pd.DataFrame:
        return self.covariance_matrix()

    def fitted(self) -> pd.Series:
        values = self.fittedvalues
        model_index = getattr(self.result.model, "data", None)
        index = getattr(model_index, "row_labels", None) if model_index is not None else None
        return pd.Series(values, index=index, name=f"FITTED({self.dependent})")

    def residual_series(self) -> pd.Series:
        values = self.residuals
        model_data = getattr(self.result, "model", None)
        index = getattr(getattr(model_data, "data", None), "row_labels", None) if model_data is not None else None
        return pd.Series(values, index=index, name=f"RESID({self.dependent})")

    @property
    def inverted_ar_roots(self) -> np.ndarray:
        roots = np.asarray(getattr(self.result, "arroots", np.array([], dtype=complex)), dtype=complex)
        if roots.size:
            return 1.0 / roots
        if self.error_process.p:
            from numpy.polynomial import polynomial
            coeff = np.zeros(self.error_process.max_p + 1)
            coeff[0] = 1.0
            for lag in self.error_process.p:
                coeff[lag] = -float(self.params[f"AR({lag})"])
            return polynomial.polyroots(coeff)[::-1]
        return roots

    @property
    def inverted_ma_roots(self) -> np.ndarray:
        roots = np.asarray(getattr(self.result, "maroots", np.array([], dtype=complex)), dtype=complex)
        if roots.size:
            return 1.0 / roots
        if self.error_process.q:
            from numpy.polynomial import polynomial
            coeff = np.zeros(self.error_process.max_q + 1)
            coeff[0] = 1.0
            for lag in self.error_process.q:
                coeff[lag] = float(self.params[f"MA({lag})"])
            return polynomial.polyroots(coeff)[::-1]
        return roots

    def arma_structure(self, type: str = "root", *, hrz: int = 25, impulse: float | None = None) -> pd.DataFrame | dict[str, np.ndarray]:
        """Replicate EViews Equation.arma views: root, acf, imp, or freq."""
        if not (self.error_process.max_p or self.error_process.max_q):
            raise ValueError("ARMA structure views require an AR or MA error term")
        kind = type.lower()
        if kind == "root":
            return self.roots_report()
        if hrz < 1:
            raise ValueError("hrz must be positive")
        from statsmodels.tsa.arima_process import ArmaProcess
        params = self.params
        ar_poly = np.zeros(self.error_process.max_p + 1)
        ma_poly = np.zeros(self.error_process.max_q + 1)
        ar_poly[0] = 1.0
        ma_poly[0] = 1.0
        for lag in self.error_process.p:
            ar_poly[lag] = -float(params[f"AR({lag})"])
        for lag in self.error_process.q:
            ma_poly[lag] = float(params[f"MA({lag})"])
        process = ArmaProcess(ar_poly, ma_poly)
        if kind == "acf":
            ac = process.acf(lags=hrz + 1)
            pac = process.pacf(lags=hrz + 1)
            return pd.DataFrame({"Lag": np.arange(ac.size), "AC": ac, "PAC": pac})
        if kind == "imp":
            shock = float(np.sqrt(params.get("SIGMASQ", np.nan))) if impulse is None else float(impulse)
            response = process.impulse_response(leads=hrz - 1) * shock
            return pd.DataFrame({"Period": np.arange(response.size), "Impulse response": response})
        if kind == "freq":
            freq, spec = process.periodogram(nobs=max(256, hrz * 16), whole=True)
            return pd.DataFrame({"Frequency": freq, "Spectrum": spec})
        raise ValueError("type must be 'root', 'acf', 'imp', or 'freq'")

    def roots_report(self) -> dict[str, np.ndarray]:
        return {"Inverted AR Roots": self.inverted_ar_roots, "Inverted MA Roots": self.inverted_ma_roots}

    def convergence_info(self) -> dict[str, Any]:
        if hasattr(self.result, "converged"):
            return {"converged": bool(self.result.converged), "iterations": int(getattr(self.result, "iterations", 0)), "optimizer": "BFGS"}
        retvals = getattr(self.result, "mle_retvals", {}) or {}
        return {"converged": retvals.get("converged"), "iterations": retvals.get("iterations", retvals.get("nit")), "optimizer": "BFGS"}

    def arma(self, type: str = "root", *, hrz: int = 25, impulse: float | None = None):
        return self.arma_structure(type=type, hrz=hrz, impulse=impulse)

    def auto(self, order: int = 1, *, alpha: float = 0.05):
        """EViews equation.auto(order): Breusch-Godfrey LM test."""
        return self.serial_correlation_lm(order, alpha=alpha)

    def hettest(self, *, type: str = "BPG", lags: int = 12, cross_terms: bool = False, alpha: float = 0.05):
        """EViews equation.hettest(type=...) diagnostic."""
        return self.heteroskedasticity(test=type, lags=lags, cross_terms=cross_terms, alpha=alpha)

    def archtest(self, lags: int = 12, *, alpha: float = 0.05):
        """EViews equation.archtest diagnostic."""
        return self.heteroskedasticity(test="ARCH", lags=lags, alpha=alpha)

    def white(self, *, cross_terms: bool = False, alpha: float = 0.05):
        """EViews equation.white diagnostic."""
        return self.heteroskedasticity(test="White", cross_terms=cross_terms, alpha=alpha)

    def histogram_normality(self, *, alpha: float = 0.05):
        """EViews Histogram-Normality residual view."""
        return self.normality_test(alpha=alpha)
    def statistics(self) -> dict[str, float]:
        """Return EViews-normalized regression statistics."""
        nobs = float(self.nobs)
        nparams = float(len(self.params))
        llf = float(getattr(self.result, "llf", np.nan))
        raw_scale = float(getattr(self.result, "scale", np.nan))
        if self.error_process.max_p or self.error_process.max_q:
            sigma2 = float(self.params.get("SIGMASQ", raw_scale))
            residuals = np.asarray(self.residuals, dtype=float)
            valid = residuals[np.isfinite(residuals)]
            ssr = float(sigma2 * nobs) if np.isfinite(sigma2) else float(np.dot(valid, valid)) if valid.size else np.nan
            sse = float(np.sqrt(sigma2)) if np.isfinite(sigma2) and sigma2 >= 0 else np.nan
            if np.isfinite(llf) and nobs > 0:
                aic = -2.0 * llf / nobs + 2.0 * nparams / nobs
                bic = -2.0 * llf / nobs + nparams * np.log(nobs) / nobs
                hqic = -2.0 * llf / nobs + 2.0 * nparams * np.log(np.log(nobs)) / nobs
            else:
                aic = bic = hqic = np.nan
            y = getattr(getattr(self.result, "model", None), "endog", None)
            y = np.asarray(y, dtype=float).reshape(-1) if y is not None else np.array([])
            fitted = np.asarray(self.fittedvalues, dtype=float)
            if y.size and fitted.size == y.size:
                r2 = 1.0 - float(np.sum((y - fitted) ** 2)) / float(np.sum((y - y.mean()) ** 2))
                adj = 1.0 - (1.0 - r2) * (nobs - 1.0) / max(nobs - len(self.params), 1.0)
            else:
                r2 = adj = np.nan
            dw_num = float(np.sum(np.diff(valid) ** 2)) if valid.size > 1 else np.nan
            dw_den = float(np.sum(valid ** 2)) if valid.size else np.nan
            return {
                "R-squared": r2,
                "Adjusted R-squared": adj,
                "S.E. of regression": sse,
                "Sum squared resid": ssr,
                "Log likelihood": llf,
                "Akaike info criterion": aic,
                "Schwarz criterion": bic,
                "Hannan-Quinn criterion": hqic,
                "Durbin-Watson": dw_num / dw_den if dw_den else np.nan,
                "F-statistic": np.nan,
                "Prob(F-statistic)": np.nan,
            }
        if np.isfinite(llf) and nobs > 0:
            aic = -2.0 * llf / nobs + 2.0 * nparams / nobs
            bic = -2.0 * llf / nobs + nparams * np.log(nobs) / nobs
            hqic = -2.0 * llf / nobs + 2.0 * nparams * np.log(np.log(nobs)) / nobs
        else:
            aic = bic = hqic = float("nan")
        return {
            "R-squared": float(getattr(self.result, "rsquared", np.nan)),
            "Adjusted R-squared": float(getattr(self.result, "rsquared_adj", np.nan)),
            "S.E. of regression": float(np.sqrt(raw_scale)) if np.isfinite(raw_scale) and raw_scale >= 0 else float("nan"),
            "Sum squared resid": float(getattr(self.result, "ssr", np.nan)),
            "Log likelihood": llf,
            "Akaike info criterion": aic,
            "Schwarz criterion": bic,
            "Hannan-Quinn criterion": hqic,
            "Durbin-Watson": float(getattr(self.result, "dw", np.nan)),
            "F-statistic": float(getattr(self.result, "fvalue", np.nan)),
            "Prob(F-statistic)": float(getattr(self.result, "f_pvalue", np.nan)),
        }

    def serial_correlation(self, lags: int = 1):
        from .diagnostics import breusch_godfrey
        return breusch_godfrey(self.result, lags=lags)


@dataclass
class Equation:
    workfile: object
    name: str = "EQ01"
    specification: str = ""
    result: EquationResult | None = None

    def _build_regressors(self, dependent: Any, tokens: list[str]) -> tuple[pd.DataFrame, ErrorProcess]:
        regressors, error_process = parse_error_terms(tokens)
        frame_index = dependent.index if dependent.index is not None else tuple(range(dependent.nobs))
        frame = pd.DataFrame(index=frame_index)
        names: list[str] = []
        for token in regressors:
            if token.upper() == "C":
                frame["C"] = 1.0
                names.append("C")
                continue
            if token.upper() == "@TREND":
                frame["@TREND"] = np.arange(dependent.nobs, dtype=float)
                names.append("@TREND")
                continue
            try:
                value = evaluate(token, self.workfile)
            except ExpressionError as exc:
                raise ValueError(f"invalid regressor {token!r}: {exc}") from exc
            if not hasattr(value, "values"):
                raise ValueError(f"regressor {token!r} did not produce a time series")
            if value.nobs != self.workfile.nobs:
                value = self.workfile._pad_to_workfile(value, name=token)
            series = value[self.workfile.sample_indexer]
            if series.nobs != dependent.nobs:
                raise ValueError(f"regressor {token!r} has incompatible length")
            frame[token] = series.values
            names.append(token)
        return frame[names], error_process

    def ls(self, specification: str | None = None, *, start_params: np.ndarray | list[float] | None = None, arma_method: str = "ml", arma_start: str = "automatic", backcast: bool = True, covariance: str = "opg", optimizer: str = "bfgs", maxiter: int = 1000, tol: float = 1e-8, random_seed: int | None = None) -> EquationResult:
        """Estimate an OLS or EViews-style ARMA-error equation using BFGS ML."""
        spec = (specification or self.specification).strip()
        if not spec:
            raise ValueError("an equation specification is required")
        tokens = _expand_eviews_ranges(spec)
        if len(tokens) < 2:
            raise ValueError("equation specification must contain a dependent variable and regressors")
        dependent_name = tokens[0]
        dependent = self.workfile.sample_series(dependent_name)
        X, error_process = self._build_regressors(dependent, tokens[1:])
        frame_index = dependent.index if dependent.index is not None else tuple(range(dependent.nobs))
        y_series = pd.Series(np.asarray(dependent.values, dtype=float), index=frame_index, name=dependent_name)
        frame = pd.concat([y_series, X], axis=1)
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        if frame.empty:
            raise ValueError("no observations remain after applying the equation sample")
        y = frame[dependent_name]
        exog = frame[X.columns]
        if error_process.max_p or error_process.max_q or error_process.max_sar or error_process.max_sma:
            method_key = arma_method.lower()
            if method_key not in {"ml", "gls", "cls"}:
                raise ValueError("arma_method must be ml, gls, or cls")
            start_mode = "user-specified" if start_params is not None else arma_start
            start = make_starting_values(
                np.asarray(y, dtype=float), np.asarray(exog, dtype=float),
                error_process.p, error_process.q, method=method_key,
                mode=start_mode, user=start_params, random_seed=random_seed,
            )
            if method_key == "cls":
                result = fit_cls(
                    np.asarray(y, dtype=float), np.asarray(exog, dtype=float),
                    list(exog.columns), error_process.p, error_process.q,
                    start_values=start, backcast=backcast, maxiter=maxiter, tol=tol,
                )
                method = "ARMA Conditional Least Squares (BFGS)"
            elif method_key == "gls":
                result = fit_gls(
                    np.asarray(y, dtype=float), np.asarray(exog, dtype=float),
                    list(exog.columns), error_process.p, error_process.q,
                    start_values=start, maxiter=maxiter, tol=tol,
                )
                method = "ARMA Generalized Least Squares (BFGS)"
            else:
                model = SARIMAX(
                    y, exog=exog,
                    order=(error_process.max_p, 0, error_process.max_q),
                    seasonal_order=(error_process.max_sar, 0, error_process.max_sma, self.workfile.seasonal_period) if (error_process.max_sar or error_process.max_sma) else (0, 0, 0, 0),
                    trend="n", enforce_stationarity=True, enforce_invertibility=True,
                )
                if covariance.lower() not in {"opg", "hessian"}:
                    raise ValueError("covariance must be opg or hessian")
                kwargs = {"method": "bfgs", "disp": False, "maxiter": maxiter}
                if covariance.lower() == "opg":
                    kwargs["cov_type"] = "opg"
                full_start = np.asarray(start, dtype=float)
                if full_start.size == len(model.param_names):
                    kwargs["start_params"] = full_start
                constraints = {}
                for lag in range(1, error_process.max_p + 1):
                    if lag not in error_process.p:
                        constraints[f"ar.L{lag}"] = 0.0
                for lag in range(1, error_process.max_q + 1):
                    if lag not in error_process.q:
                        constraints[f"ma.L{lag}"] = 0.0
                if constraints:
                    result = model.fit_constrained(constraints, **kwargs)
                else:
                    result = model.fit(**kwargs)
                method = "ARMA Maximum Likelihood (BFGS)"
        else:
            model = sm.OLS(y, exog)
            result = model.fit()
            method = "Least Squares"
        opg = None
        if error_process.max_p or error_process.max_q:
            try:
                score_obs = np.asarray(result.model.score_obs(result.params), dtype=float)
                info = score_obs.T @ score_obs
                opg = pd.DataFrame(np.linalg.pinv(info), index=_rename_arma_parameters(pd.Series(result.params, index=result.param_names), error_process).index, columns=_rename_arma_parameters(pd.Series(result.params, index=result.param_names), error_process).index)
            except Exception:
                opg = None
        sample = f"{frame.index[0]} {frame.index[-1]}" if len(frame.index) else None
        wrapped = EquationResult(result=result, title=f"Equation: {self.name}", dependent=dependent_name, method=method, sample=sample, specification=spec, error_process=error_process, _opg_covariance=opg, observed=np.asarray(y, dtype=float), workfile=self.workfile)
        self.specification = spec
        self.result = wrapped
        return wrapped

    def estimate(
        self, method: str = "LS", specification: str | None = None,
        *, start_params: np.ndarray | list[float] | None = None,
        arma_start: str = "automatic", backcast: bool = True,
        covariance: str = "opg", optimizer: str = "bfgs",
        maxiter: int = 1000, tol: float = 1e-8, random_seed: int | None = None,
    ) -> EquationResult:
        """Estimate using an EViews-compatible equation method."""
        method_upper = method.upper().replace(" ", "")
        if method_upper in {"LS", "OLS", "MCO"}:
            return self.ls(specification)
        if method_upper in {"ML", "ARMA", "ARMAX"}:
            arma_method = "ml"
        elif method_upper == "CLS":
            arma_method = "cls"
        elif method_upper == "GLS":
            arma_method = "gls"
        else:
            raise ValueError("method must be LS/OLS, ML, CLS, or GLS")
        return self.ls(
            specification, start_params=start_params, arma_method=arma_method,
            arma_start=arma_start, backcast=backcast, covariance=covariance,
            optimizer=optimizer, maxiter=maxiter, tol=tol, random_seed=random_seed,
        )

    def show(self, *, view: str = "estimate") -> str:
        if self.result is None:
            return f"Equation {self.name}: {self.specification or '(not estimated)'}"
        if view.lower() in {"estimate", "output", "summary"}:
            return self.result.summary()
        return self.result.view_text(view)
