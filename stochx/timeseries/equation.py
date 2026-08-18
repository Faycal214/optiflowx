"""EViews-inspired equation specifications and estimation objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .expression import ExpressionError, evaluate
from .results import UnifiedResult


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
        tokens = spec.split()
        if len(tokens) < 2:
            raise ValueError("OLS specification must contain a dependent variable and at least one regressor")
        dependent_name = tokens[0]
        dependent = self.workfile.sample_series(dependent_name)
        regressors: list[pd.Series] = []
        names: list[str] = []
        for token in tokens[1:]:
            if token.upper() == "C":
                regressors.append(pd.Series(np.ones(dependent.nobs), index=dependent.index))
                names.append("C")
                continue
            try:
                value = evaluate(token, self.workfile)
            except ExpressionError as exc:
                raise ValueError(f"invalid regressor {token!r}: {exc}") from exc
            if not hasattr(value, "values"):
                raise ValueError(f"regressor {token!r} did not produce a time series")
            series = value
            if series.nobs != dependent.nobs:
                raise ValueError(f"regressor {token!r} has incompatible length")
            regressors.append(pd.Series(series.values, index=series.index))
            names.append(token)

        frame = pd.DataFrame({dependent_name: dependent.values}, index=dependent.index)
        for name, regressor in zip(names, regressors):
            frame[name] = regressor.to_numpy(dtype=float)
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        if frame.empty:
            raise ValueError("no observations remain after applying the equation sample")
        y = frame[dependent_name].to_numpy(dtype=float)
        X = frame[names].to_numpy(dtype=float)
        if "C" in names:
            # C is already an explicit constant column.
            pass
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
