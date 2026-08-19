"""EViews-inspired equation specifications and estimation objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .arma_errors import ErrorProcess, parse_error_terms
from .expression import ExpressionError, evaluate
from .results import UnifiedResult

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
        elif text.lower() in {"sigma2", "sigmasq"}:
            mapping[text] = "SIGMASQ"
    return series.rename(index=mapping)


@dataclass
class EquationResult(UnifiedResult):
    """OLS or ARMA-error equation result with EViews-style output."""

    specification: str = ""
    error_process: ErrorProcess = ErrorProcess()

    def _parameter_series(self, attribute: str) -> pd.Series:
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
    def covariance_method(self) -> str:
        """Return the coefficient covariance method used for this result."""
        return "outer product of gradients (OPG)" if (self.error_process.max_p or self.error_process.max_q) else "OLS inverse information"

    def covariance_matrix(self) -> pd.DataFrame:
        """Return the EViews-style coefficient covariance matrix.

        For ARMA maximum-likelihood equations EViews reports coefficient
        covariance computed from the outer product of per-observation score
        vectors. StochX follows the same convention and does not apply an
        additional finite-sample scaling factor; this matches the published
        EViews tutorial results used by the Phase B parity fixture.
        """
        if not (self.error_process.max_p or self.error_process.max_q):
            cov = np.asarray(self.result.cov_params(), dtype=float)
            names = list(self.params.index)
            return pd.DataFrame(cov, index=names, columns=names)

        model = self.result.model
        params = np.asarray(self.result.params, dtype=float)
        score_obs = np.asarray(model.score_obs(params, approx_complex_step=False), dtype=float)
        if score_obs.ndim != 2 or score_obs.shape[1] != params.size:
            raise RuntimeError("ARMA score matrix has an unexpected shape")
        information = score_obs.T @ score_obs
        covariance = np.linalg.pinv(information, rcond=1e-12)
        names = list(self.params.index)
        return pd.DataFrame(covariance, index=names, columns=names)

    @property
    def covariance(self) -> pd.DataFrame:
        """Alias for :meth:`covariance_matrix`."""
        return self.covariance_matrix()

    def _arma_bse(self) -> pd.Series:
        covariance = self.covariance_matrix()
        diagonal = np.maximum(np.diag(covariance.to_numpy(dtype=float)), 0.0)
        return pd.Series(np.sqrt(diagonal), index=covariance.index, dtype=float)

    @property
    def bse(self) -> pd.Series:
        if self.error_process.max_p or self.error_process.max_q:
            return self._arma_bse()
        return self._parameter_series("bse").reindex(self.params.index)

    @property
    def tvalues(self) -> pd.Series:
        if self.error_process.max_p or self.error_process.max_q:
            return self.params / self.bse
        return self._parameter_series("tvalues").reindex(self.params.index)

    @property
    def pvalues(self) -> pd.Series:
        if self.error_process.max_p or self.error_process.max_q:
            df_resid = max(int(self.nobs - len(self.params)), 1)
            values = 2.0 * stats.t.sf(np.abs(self.tvalues.to_numpy(dtype=float)), df=df_resid)
            return pd.Series(values, index=self.params.index, dtype=float)
        return self._parameter_series("pvalues").reindex(self.params.index)

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
        """Return EViews-style inverse AR characteristic roots."""
        roots = np.asarray(getattr(self.result, "arroots", np.array([], dtype=complex)), dtype=complex)
        return 1.0 / roots if roots.size else roots

    @property
    def inverted_ma_roots(self) -> np.ndarray:
        """Return EViews-style inverse MA characteristic roots."""
        roots = np.asarray(getattr(self.result, "maroots", np.array([], dtype=complex)), dtype=complex)
        return 1.0 / roots if roots.size else roots

    def roots_report(self) -> dict[str, np.ndarray]:
        return {"Inverted AR Roots": self.inverted_ar_roots, "Inverted MA Roots": self.inverted_ma_roots}

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
    """An equation belonging to a StochX Workfile."""

    workfile: object
    name: str = "EQ01"
    specification: str = ""
    result: EquationResult | None = None

    def _build_regressors(self, dependent: Any, tokens: list[str]) -> tuple[pd.DataFrame, ErrorProcess]:
        regressors, error_process = parse_error_terms(tokens)
        frame = pd.DataFrame(index=dependent.index)
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
            series = value[self.workfile.sample]
            if series.nobs != dependent.nobs:
                raise ValueError(f"regressor {token!r} has incompatible length")
            frame[token] = series.values
            names.append(token)
        return frame[names], error_process

    def ls(self, specification: str | None = None, *, start_params: np.ndarray | list[float] | None = None) -> EquationResult:
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
        frame = pd.concat([dependent.astype(float).rename(dependent_name), X], axis=1)
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        if frame.empty:
            raise ValueError("no observations remain after applying the equation sample")
        y = frame[dependent_name]
        exog = frame[X.columns]
        if error_process.max_p or error_process.max_q:
            model = SARIMAX(
                y,
                exog=exog,
                order=(error_process.max_p, 0, error_process.max_q),
                trend="n",
                enforce_stationarity=True,
                enforce_invertibility=True,
            )
            kwargs = {"method": "bfgs", "disp": False, "maxiter": 1000}
            if start_params is not None:
                kwargs["start_params"] = np.asarray(start_params, dtype=float)
            result = model.fit(**kwargs)
            method = "ARMA Maximum Likelihood (BFGS)"
        else:
            model = sm.OLS(y, exog)
            result = model.fit()
            method = "Least Squares"
        sample = f"{frame.index[0]} {frame.index[-1]}" if len(frame.index) else None
        wrapped = EquationResult(
            result=result,
            title=f"Equation: {self.name}",
            dependent=dependent_name,
            method=method,
            sample=sample,
            specification=spec,
            error_process=error_process,
        )
        self.specification = spec
        self.result = wrapped
        return wrapped

    def estimate(self, method: str = "LS", specification: str | None = None, *, start_params: np.ndarray | list[float] | None = None) -> EquationResult:
        """Estimate using LS/OLS or ARMA maximum likelihood error terms."""
        method_upper = method.upper().replace(" ", "")
        if method_upper in {"LS", "OLS", "MCO", "ML", "ARMA", "ARMAX"}:
            return self.ls(specification, start_params=start_params)
        raise NotImplementedError("Equation supports LS/OLS and ARMA-error maximum likelihood")

    def show(self) -> str:
        if self.result is None:
            return f"Equation {self.name}: {self.specification or '(not estimated)'}"
        return self.result.summary()
