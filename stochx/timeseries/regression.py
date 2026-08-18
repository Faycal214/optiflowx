"""Deterministic regression tools used in the course's time-series analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .series import TimeSeries


def _as_array(y) -> np.ndarray:
    x = np.asarray(y.values if isinstance(y, TimeSeries) else list(y), dtype=float).reshape(-1)
    return x[np.isfinite(x)]


@dataclass
class RegressionResult:
    """OLS result with an EViews-style report."""

    result: object
    dependent_name: str = "Y"

    def summary(self) -> str:
        """Return the underlying regression report."""
        return str(self.result.summary())

    @property
    def params(self):
        """Return estimated regression coefficients."""
        return self.result.params

    @property
    def residuals(self):
        """Return regression residuals."""
        return np.asarray(self.result.resid, dtype=float)


def ols(y, X: Iterable[Iterable[float]] | None = None, *, trend: str | None = "c", name: str = "Y") -> RegressionResult:
    """Estimate an OLS deterministic regression with optional constant/trend."""
    target = _as_array(y)
    if X is None:
        t = np.arange(1, target.size + 1, dtype=float)
        if trend == "ct":
            exog = np.column_stack([np.ones(target.size), t])
        elif trend == "c":
            exog = np.ones((target.size, 1))
        elif trend == "n" or trend is None:
            exog = np.empty((target.size, 0))
        else:
            raise ValueError("trend must be 'n', 'c', or 'ct'")
    else:
        exog = np.asarray(list(X), dtype=float)
        if exog.ndim == 1:
            exog = exog[:, None]
    if exog.shape[1] and trend not in {"n", None} and X is not None:
        exog = sm.add_constant(exog, has_constant="add") if trend == "c" else sm.add_constant(np.column_stack([np.arange(target.size), exog]), has_constant="add")
    result = sm.OLS(target, exog if exog.shape[1] else np.empty((target.size, 0))).fit()
    return RegressionResult(result, name)


def trend_terms(n: int, *, degree: int = 1) -> pd.DataFrame:
    """Create intercept and polynomial time-trend regressors."""
    if n < 1 or degree < 0:
        raise ValueError("n must be positive and degree non-negative")
    t = np.arange(1, n + 1, dtype=float)
    frame = {"const": np.ones(n)}
    for d in range(1, degree + 1):
        frame[f"trend{d}"] = t**d
    return pd.DataFrame(frame)
