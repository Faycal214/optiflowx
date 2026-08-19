"""EViews-inspired equation specifications and estimation objects."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .expression import ExpressionError, evaluate
from .results import UnifiedResult


_RANGE_RE = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\(\s*(?P<start>-?\d+)\s+to\s+(?P<end>-?\d+)\s*\)$",
    re.IGNORECASE,
)


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


@dataclass
class EquationResult(UnifiedResult):
    """OLS equation result with EViews-style output and interpretation."""

    specification: str = ""

    def fitted(self) -> pd.Series:
        """Return fitted values preserving the model's index when available."""
        values = self.fittedvalues
        model_index = getattr(self.result.model, "data", None)
        index = getattr(model_index, "row_labels", None) if model_index is not None else None
        return pd.Series(values, index=index, name=f"FITTED({self.dependent})")

    def residual_series(self) -> pd.Series:
        """Return residuals as a labelled pandas Series."""
        values = self.residuals
        model_data = getattr(self.result, "model", None)
        index = getattr(getattr(model_data, "data", None), "row_labels", None) if model_data is not None else None
        return pd.Series(values, index=index, name=f"RESID({self.dependent})")

    def statistics(self) -> dict[str, float]:
        """Return EViews-normalized regression statistics."""
        nobs = float(self.nobs)
        nparams = float(len(self.params))
        llf = float(getattr(self.result, "llf", np.nan))
        scale = float(getattr(self.result, "scale", np.nan))
        if np.isfinite(llf) and nobs > 0:
            aic = -2.0 * llf / nobs + 2.0 * nparams / nobs
            bic = -2.0 * llf / nobs + nparams * np.log(nobs) / nobs
            hqic = -2.0 * llf / nobs + 2.0 * nparams * np.log(np.log(nobs)) / nobs
        else:
            aic = bic = hqic = float("nan")
        return {
            "R-squared": float(getattr(self.result, "rsquared", np.nan)),
            "Adjusted R-squared": float(getattr(self.result, "rsquared_adj", np.nan)),
            "S.E. of regression": float(np.sqrt(scale)) if np.isfinite(scale) and scale >= 0 else float("nan"),
            "Sum squared resid": float(getattr(self.result, "ssr", np.nan)),
            "Log likelihood": llf,
            "Akaike info criterion": aic,
            "Schwarz criterion": bic,
            "Hannan-Quinn criterion": hqic,
            "Durbin-Watson": float(getattr(self.result, "dw", np.nan)),
            "F-statistic": float(getattr(self.result, "fvalue", np.nan)),
            "Prob(F-statistic)": float(getattr(self.result, "f_pvalue", np.nan)),
        }


@dataclass
class Equation:
    """An equation belonging to a StochX Workfile."""

    workfile: object
    name: str = "EQ01"
    specification: str = ""
    result: EquationResult | None = None

    def ls(self, specification: str | None = None) -> EquationResult:
        """Estimate an OLS equation using EViews-like ``Y C X(-1) Z`` syntax."""
        spec = (specification or self.specification).strip()
        if not spec:
            raise ValueError("an equation specification is required")
        tokens = _expand_eviews_ranges(spec)
        if len(tokens) < 2:
            raise ValueError("OLS specification must contain a dependent variable and at least one regressor")
        dependent_name = tokens[0]
        dependent = self.workfile.sample_series(dependent_name)
        regressors: list[pd.Series] = []
        names: list[str] = []
        for token in tokens[1:]:
            if token.upper() == "C":
                regressors.append(pd.Series(np.ones(dependent.nobs), index=dependent.index, name="C"))
                names.append("C")
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
            regressors.append(pd.Series(series.values, index=series.index, name=token))
            names.append(token)

        frame = pd.DataFrame({dependent_name: dependent.values}, index=dependent.index)
        for name, regressor in zip(names, regressors):
            frame[name] = regressor.to_numpy(dtype=float)
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        if frame.empty:
            raise ValueError("no observations remain after applying the equation sample")
        y = frame[dependent_name]
        X = frame[names]
        model = sm.OLS(y, X)
        result = model.fit()
        sample = f"{frame.index[0]} {frame.index[-1]}" if frame.index is not None and len(frame.index) else None
        wrapped = EquationResult(
            result=result,
            title=f"Equation: {self.name}",
            dependent=dependent_name,
            method="Least Squares",
            sample=sample,
            specification=spec,
        )
        self.specification = spec
        self.result = wrapped
        return wrapped

    def estimate(self, method: str = "LS", specification: str | None = None) -> EquationResult:
        """Estimate the equation using the supported EViews-style method name."""
        method_upper = method.upper().replace(" ", "")
        if method_upper in {"LS", "OLS", "MCO"}:
            return self.ls(specification)
        raise NotImplementedError("Equation currently supports LS/OLS; ARMA-family models are available through estimate()")

    def show(self) -> str:
        """Return the current estimation report."""
        if self.result is None:
            return f"Equation {self.name}: {self.specification or '(not estimated)'}"
        return self.result.summary()
