"""Unified result objects and EViews-style tables for StochX time series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResultTable:
    """Named result table with deterministic formatting."""

    data: pd.DataFrame
    title: str = ""

    def dataframe(self) -> pd.DataFrame:
        """Return a defensive copy of the result table."""
        return self.data.copy()

    def text(self, *, float_format: str = ".6f") -> str:
        """Render the table in a compact EViews-like layout."""
        formatter = lambda x: f"{x:{float_format}}" if isinstance(x, (int, float, np.floating)) else str(x)
        body = self.data.to_string(float_format=formatter)
        return f"{self.title}\n{body}" if self.title else body


@dataclass
class UnifiedResult:
    """Common public interface for fitted econometric/statistical results."""

    result: Any
    title: str
    dependent: str = "Y"
    method: str = ""
    sample: str | None = None

    @property
    def nobs(self) -> int:
        """Return the number of observations used by the fitted result."""
        value = getattr(self.result, "nobs", np.nan)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @property
    def params(self) -> pd.Series:
        """Return estimated parameters."""
        return _as_series(getattr(self.result, "params", pd.Series(dtype=float)))

    @property
    def bse(self) -> pd.Series:
        """Return parameter standard errors."""
        return _as_series(getattr(self.result, "bse", pd.Series(index=self.params.index, dtype=float)), self.params.index)

    @property
    def tvalues(self) -> pd.Series:
        """Return parameter t-statistics."""
        return _as_series(getattr(self.result, "tvalues", pd.Series(index=self.params.index, dtype=float)), self.params.index)

    @property
    def pvalues(self) -> pd.Series:
        """Return parameter p-values."""
        return _as_series(getattr(self.result, "pvalues", pd.Series(index=self.params.index, dtype=float)), self.params.index)

    @property
    def residuals(self) -> np.ndarray:
        """Return fitted residuals."""
        return np.asarray(getattr(self.result, "resid", np.array([], dtype=float)), dtype=float)

    @property
    def fittedvalues(self) -> np.ndarray:
        """Return fitted values."""
        return np.asarray(getattr(self.result, "fittedvalues", np.array([], dtype=float)), dtype=float)

    def coefficient_table(self) -> ResultTable:
        """Return coefficients, standard errors, t-statistics and p-values."""
        frame = pd.DataFrame(
            {
                "Coefficient": self.params,
                "Std. Error": self.bse,
                "t-Statistic": self.tvalues,
                "Prob.": self.pvalues,
            }
        )
        return ResultTable(frame, "Variable")

    def statistics(self) -> dict[str, float]:
        """Return model statistics using EViews' reported scaling conventions."""
        n = max(self.nobs, 1)
        k = len(self.params)
        llf = float(getattr(self.result, "llf", np.nan))
        scale = float(getattr(self.result, "scale", np.nan))
        mapping = {
            "R-squared": getattr(self.result, "rsquared", np.nan),
            "Adjusted R-squared": getattr(self.result, "rsquared_adj", np.nan),
            "S.E. of regression": np.sqrt(scale) if np.isfinite(scale) else np.nan,
            "Sum squared resid": getattr(self.result, "ssr", np.nan),
            "Log likelihood": llf,
            "Akaike info criterion": (-2.0 * llf + 2.0 * k) / n if np.isfinite(llf) else np.nan,
            "Schwarz criterion": (-2.0 * llf + k * np.log(n)) / n if np.isfinite(llf) else np.nan,
            "Hannan-Quinn criter.": (-2.0 * llf + 2.0 * k * np.log(np.log(n))) / n if np.isfinite(llf) and n > 1 else np.nan,
            "Durbin-Watson stat": getattr(self.result, "dw", np.nan),
            "F-statistic": getattr(self.result, "fvalue", np.nan),
            "Prob(F-statistic)": getattr(self.result, "f_pvalue", np.nan),
        }
        return {label: float(value) if np.isscalar(value) else np.nan for label, value in mapping.items()}
    def eviews_statistics(self) -> dict[str, float]:
        values = dict(self.statistics())
        y = getattr(getattr(self.result, "model", None), "endog", None)
        if y is not None:
            y = np.asarray(y, dtype=float).reshape(-1)
            y = y[np.isfinite(y)]
            if y.size:
                values["Mean dependent var"] = float(np.mean(y))
                values["S.D. dependent var"] = float(np.std(y, ddof=1)) if y.size > 1 else float("nan")
        return values

    def table(self) -> pd.DataFrame:
        """Return the coefficient table as a DataFrame."""
        return self.coefficient_table().dataframe()

    def summary(self) -> str:
        """Render a deterministic EViews-style estimation report."""
        lines = [self.title, "=" * 72]
        if self.method:
            lines.append(f"Method: {self.method}")
        lines.append(f"Dependent Variable: {self.dependent}")
        if self.sample:
            lines.append(f"Sample: {self.sample}")
        lines.append(f"Included observations: {self.nobs}")
        lines.append("")
        lines.append(self.coefficient_table().text())
        lines.append("")
        lines.append("Model statistics")
        for label, value in self.statistics().items():
            if np.isfinite(value):
                lines.append(f"{label:24s} {value: .6f}")
        return "\n".join(lines)

    def interpret(self, alpha: float = 0.05) -> str:
        """Produce a course-oriented interpretation of parameter significance and fit."""
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between 0 and 1")
        pvalues = self.pvalues
        statements: list[str] = []
        for name, pvalue in pvalues.items():
            if np.isfinite(pvalue):
                significance = "statistically significant" if pvalue < alpha else "not statistically significant"
                statements.append(f"{name} is {significance} at the {alpha:.0%} level (p={pvalue:.4g}).")
        if not statements:
            statements.append("No parameter significance information is available.")
        stats = self.statistics()
        aic = stats.get("Akaike info criterion", np.nan)
        bic = stats.get("Schwarz criterion", np.nan)
        if np.isfinite(aic) or np.isfinite(bic):
            statements.append("For model selection, compare AIC and Schwarz/BIC across competing specifications; smaller values are preferred.")
        dw = stats.get("Durbin-Watson stat", np.nan)
        if np.isfinite(dw):
            if dw < 1.5:
                statements.append(f"Durbin-Watson={dw:.3f} indicates potential positive residual autocorrelation and requires diagnostic checking.")
            elif dw > 2.5:
                statements.append(f"Durbin-Watson={dw:.3f} indicates potential negative residual autocorrelation and requires diagnostic checking.")
            else:
                statements.append(f"Durbin-Watson={dw:.3f} does not by itself indicate strong first-order residual autocorrelation.")
        statements.append("Residual diagnostics should be checked before accepting the specification for forecasting.")
        return " ".join(statements)


def _as_series(value: Any, index: Iterable[Any] | None = None) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.astype(float)
    array = np.asarray(value, dtype=float).reshape(-1)
    return pd.Series(array, index=index)
