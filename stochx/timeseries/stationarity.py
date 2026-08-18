"""Stationarity, unit-root, TS/DS and differencing diagnostics.

The DF/ADF workflow follows the course convention of testing the three
possible deterministic specifications separately and making decisions from
the regression-specific Dickey-Fuller critical values rather than ordinary
Student-t / normal critical values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from .series import TimeSeries


DFRegression = Literal["n", "c", "ct"]


DF_SPECIFICATIONS: dict[str, dict[str, str]] = {
    "n": {
        "label": "Model 1 — No constant, no trend",
        "equation": "ΔYₜ = γYₜ₋₁ + ΣφᵢΔYₜ₋ᵢ + εₜ",
        "deterministic_terms": "none",
    },
    "c": {
        "label": "Model 2 — Constant, no trend",
        "equation": "ΔYₜ = α + γYₜ₋₁ + ΣφᵢΔYₜ₋ᵢ + εₜ",
        "deterministic_terms": "constant",
    },
    "ct": {
        "label": "Model 3 — Constant and deterministic trend",
        "equation": "ΔYₜ = α + βt + γYₜ₋₁ + ΣφᵢΔYₜ₋ᵢ + εₜ",
        "deterministic_terms": "constant + trend",
    },
}


@dataclass(frozen=True)
class UnitRootResult:
    """Unified EViews-style unit-root result.

    The unit-root decision is based on the deterministic-specification-specific
    Dickey-Fuller critical value.  The reported ``pvalue`` is informational;
    it is not used as a replacement for the non-standard DF critical-value
    decision rule taught in the course.
    """

    test: str
    statistic: float
    pvalue: float | None
    critical_values: dict[str, float]
    regression: DFRegression
    lags: int
    nobs: int
    null_hypothesis: str
    alternative_hypothesis: str
    decision: Literal["reject", "fail_to_reject"]
    alpha: float
    conclusion: str
    critical_value_source: str = "Regression-specific Dickey-Fuller critical values"
    specification_label: str = ""

    @property
    def rejects_null(self) -> bool:
        """Whether the unit-root null is rejected at the requested level."""
        return self.decision == "reject"

    @property
    def decision_rule(self) -> str:
        """Return the exact inequality used for the DF/ADF decision."""
        critical = self.critical_values[self._level_key]
        return f"Reject H0 when test statistic < {critical:.6f}."

    @property
    def _level_key(self) -> str:
        """Return the critical-value key corresponding to the decision level."""
        mapping = {0.01: "1%", 0.05: "5%", 0.10: "10%"}
        return min(mapping, key=lambda value: abs(value - self.alpha))

    @property
    def critical_value(self) -> float:
        """Return the critical value used by the decision."""
        return float(self.critical_values[self._level_key])

    def table(self) -> pd.DataFrame:
        """Return the unified single-test result table."""
        return pd.DataFrame(
            [
                {
                    "Test": self.test,
                    "Specification": self.specification_label or self.regression,
                    "Regression": self.regression,
                    "Lagged differences": self.lags,
                    "Observations": self.nobs,
                    "Test Statistic": self.statistic,
                    "Prob.*": self.pvalue,
                    "Critical Value": self.critical_value,
                    "Decision": "Reject H0" if self.rejects_null else "Do not reject H0",
                }
            ]
        )

    def summary(self) -> str:
        """Render an EViews-style hypothesis/decision report."""
        critical_key = self._level_key
        lines = [
            f"{self.test}",
            "=" * 76,
            f"{self.specification_label or self.regression}",
            f"Deterministic terms: {DF_SPECIFICATIONS[self.regression]['deterministic_terms']}",
            f"Test equation: {DF_SPECIFICATIONS[self.regression]['equation']}",
            f"Included observations: {self.nobs}",
            f"Lagged differences: {self.lags}",
            "",
            f"Null hypothesis (H0): {self.null_hypothesis}",
            f"Alternative (H1): {self.alternative_hypothesis}",
            "",
            f"Test statistic: {self.statistic:.6f}",
        ]
        if self.pvalue is not None and np.isfinite(self.pvalue):
            lines.append(f"Prob.*: {self.pvalue:.6f}  [informational; not the decision rule]")
        lines.extend(
            [
                "",
                "Dickey-Fuller critical values:",
                *[f"{level}: {value:.6f}" for level, value in self.critical_values.items()],
                "",
                f"Decision level: {critical_key}",
                f"Decision rule: {self.decision_rule}",
                f"Critical-value source: {self.critical_value_source}",
                f"Decision: {'Reject H0' if self.rejects_null else 'Do not reject H0'}",
                f"Conclusion: {self.conclusion}",
            ]
        )
        return "\n".join(lines)

    def interpret(self) -> str:
        """Return the course-oriented interpretation of the test decision."""
        return self.conclusion


@dataclass(frozen=True)
class SequentialDFResult:
    """Structured result for the course's Model 3 → Model 2 → Model 1 workflow."""

    tests: tuple[UnitRootResult, ...]
    selected: UnitRootResult
    nature: str
    selection_rule: str

    @property
    def rejected_at_selected_specification(self) -> bool:
        """Whether the selected specification rejects the unit-root null."""
        return self.selected.rejects_null

    def table(self) -> pd.DataFrame:
        """Return a comparison table for all DF/ADF deterministic specifications."""
        rows = []
        for result in self.tests:
            rows.append(
                {
                    "Model": result.specification_label,
                    "Regression": result.regression,
                    "Deterministic terms": DF_SPECIFICATIONS[result.regression]["deterministic_terms"],
                    "Lagged differences": result.lags,
                    "Observations": result.nobs,
                    "ADF Statistic": result.statistic,
                    "Prob.*": result.pvalue,
                    "1% CV": result.critical_values.get("1%", np.nan),
                    "5% CV": result.critical_values.get("5%", np.nan),
                    "10% CV": result.critical_values.get("10%", np.nan),
                    "Decision": "Reject H0" if result.rejects_null else "Do not reject H0",
                }
            )
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Render an EViews-style sequential DF/ADF report."""
        lines = [
            "Dickey-Fuller / Augmented Dickey-Fuller Sequential Test",
            "=" * 76,
            "Course workflow: Model 3 → Model 2 → Model 1",
            "",
            self.table().to_string(index=False, float_format=lambda value: f"{value:.6f}"),
            "",
            f"Selected specification: {self.selected.specification_label}",
            f"Selection rule: {self.selection_rule}",
            f"Series classification: {self.nature}",
            f"Interpretation: {self.selected.conclusion}",
        ]
        return "\n".join(lines)

    def interpret(self) -> str:
        """Return the final course-oriented sequential interpretation."""
        return (
            f"{self.selected.conclusion} Selected specification: "
            f"{self.selected.specification_label}. Series classification: {self.nature}."
        )


def _values(y: TimeSeries | Iterable[float]) -> np.ndarray:
    x = np.asarray(y.values if isinstance(y, TimeSeries) else list(y), dtype=float).reshape(-1)
    if np.any(np.isinf(x)):
        raise ValueError("series must contain no infinite values")
    x = x[~np.isnan(x)]
    if x.size < 10:
        raise ValueError("at least 10 observations are recommended for unit-root testing")
    return x


def _validate_alpha(alpha: float) -> None:
    if alpha not in {0.01, 0.05, 0.10}:
        raise ValueError("alpha must be one of the non-standard DF levels: 0.01, 0.05, or 0.10")


def _decision(statistic: float, critical_values: dict[str, float], alpha: float) -> Literal["reject", "fail_to_reject"]:
    _validate_alpha(alpha)
    level = {0.01: "1%", 0.05: "5%", 0.10: "10%"}[alpha]
    return "reject" if statistic < float(critical_values[level]) else "fail_to_reject"


def adf(
    y: TimeSeries | Iterable[float],
    *,
    regression: DFRegression = "ct",
    lags: int | None = None,
    autolag: str | None = "AIC",
    alpha: float = 0.05,
) -> UnitRootResult:
    """Run an ADF test under one of the course's three deterministic models.

    ``n`` is Model 1 (no constant/no trend), ``c`` is Model 2 (constant),
    and ``ct`` is Model 3 (constant + deterministic trend).  Decisions use
    the regression-specific Dickey-Fuller critical values returned for that
    specification; ordinary Student-t/normal critical values are not used.
    """
    x = _values(y)
    if regression not in DF_SPECIFICATIONS:
        raise ValueError("regression must be 'n', 'c', or 'ct'")
    if lags is not None and (not isinstance(lags, int) or lags < 0):
        raise ValueError("lags must be a non-negative integer or None")
    _validate_alpha(alpha)

    result = adfuller(x, regression=regression, maxlag=lags, autolag=autolag)
    if len(result) == 5:
        statistic, pvalue, usedlag, nobs, critical = result
    elif len(result) == 6:
        statistic, pvalue, usedlag, nobs, critical, _ = result
    else:
        raise RuntimeError(f"unexpected statsmodels ADF result length: {len(result)}")

    critical_values = {str(key): float(value) for key, value in critical.items()}
    decision = _decision(float(statistic), critical_values, alpha)
    if decision == "reject":
        conclusion = (
            f"Reject H0 at {int(alpha * 100)}%. There is evidence against a unit root "
            f"under the {DF_SPECIFICATIONS[regression]['label'].lower()} specification."
        )
    else:
        conclusion = (
            f"Do not reject H0 at {int(alpha * 100)}%. The unit-root null remains plausible "
            f"under the {DF_SPECIFICATIONS[regression]['label'].lower()} specification."
        )

    return UnitRootResult(
        test="Augmented Dickey-Fuller Test",
        statistic=float(statistic),
        pvalue=float(pvalue),
        critical_values=critical_values,
        regression=regression,
        lags=int(usedlag),
        nobs=int(nobs),
        null_hypothesis="γ = 0 (the series has a unit root / is non-stationary under the selected specification).",
        alternative_hypothesis="γ < 0 (the series is stationary under the selected specification).",
        decision=decision,
        alpha=alpha,
        conclusion=conclusion,
        specification_label=DF_SPECIFICATIONS[regression]["label"],
    )


def dickey_fuller_sequential(
    y: TimeSeries | Iterable[float],
    *,
    max_lags: int | None = None,
    autolag: str | None = "AIC",
    alpha: float = 0.05,
) -> SequentialDFResult:
    """Apply the course's sequential Model 3 → Model 2 → Model 1 DF/ADF workflow.

    The procedure evaluates the trend/intercept specification first. If the
    unit-root null is rejected there, Model 3 is selected. Otherwise the test
    proceeds to Model 2, then Model 1. Every step keeps its own regression-
    specific non-standard Dickey-Fuller critical values.
    """
    _validate_alpha(alpha)
    models = (
        adf(y, regression="ct", lags=max_lags, autolag=autolag, alpha=alpha),
        adf(y, regression="c", lags=max_lags, autolag=autolag, alpha=alpha),
        adf(y, regression="n", lags=max_lags, autolag=autolag, alpha=alpha),
    )

    selected = models[-1]
    nature = "difference-stationary / integrated candidate (DS)"
    for result in models:
        if result.rejects_null:
            selected = result
            if result.regression == "ct":
                nature = "stationary around a deterministic trend (TS candidate)"
            elif result.regression == "c":
                nature = "stationary around a constant (TS candidate)"
            else:
                nature = "stationary around zero"
            break

    return SequentialDFResult(
        tests=models,
        selected=selected,
        nature=nature,
        selection_rule=(
            "Start with Model 3 (constant + trend); if H0 is not rejected, proceed to Model 2; "
            "if H0 is still not rejected, proceed to Model 1. At each step, use that specification's "
            "Dickey-Fuller critical values rather than ordinary t critical values."
        ),
    )


def kpss_test(
    y: TimeSeries | Iterable[float],
    *,
    regression: str = "c",
    nlags: str | int = "auto",
    alpha: float = 0.05,
) -> UnitRootResult:
    """Run the KPSS stationarity test as a complementary diagnostic."""
    x = _values(y)
    statistic, pvalue, lags, critical = kpss(x, regression=regression, nlags=nlags)
    critical_values = {str(k): float(v) for k, v in critical.items()}
    _validate_alpha(alpha)
    level = {0.01: "1%", 0.05: "5%", 0.10: "10%"}[alpha]
    decision = "reject" if statistic > critical_values[level] else "fail_to_reject"
    conclusion = (
        f"Reject stationarity at {int(alpha * 100)}%; evidence favors non-stationarity."
        if decision == "reject"
        else f"Do not reject stationarity at {int(alpha * 100)}%."
    )
    return UnitRootResult(
        "KPSS Test",
        float(statistic),
        float(pvalue),
        critical_values,
        regression if regression in {"n", "c", "ct"} else "c",
        int(lags),
        int(x.size - lags),
        "The series is stationary.",
        "The series is non-stationary.",
        decision,
        alpha,
        conclusion,
        critical_value_source="KPSS critical values",
        specification_label=f"KPSS regression={regression}",
    )


def phillips_perron(
    y: TimeSeries | Iterable[float],
    *,
    trend: str = "c",
    lags: int | None = None,
) -> UnitRootResult:
    """Run Phillips-Perron when the optional ``arch`` backend is installed."""
    x = _values(y)
    if trend not in {"n", "c", "ct"}:
        raise ValueError("trend must be 'n', 'c', or 'ct'")
    try:
        from arch.unitroot import PhillipsPerron
    except ImportError as exc:
        raise ImportError("Phillips-Perron requires the optional 'arch' dependency. Install with: pip install arch") from exc
    test = PhillipsPerron(x, trend=trend, lags=lags)
    critical = {str(k): float(v) for k, v in test.critical_values.items()}
    return UnitRootResult(
        "Phillips-Perron Test",
        float(test.stat),
        float(test.pvalue),
        critical,
        trend,
        int(test.lags),
        int(test.nobs),
        "The series contains a unit root.",
        "The series is stationary.",
        _decision(float(test.stat), critical, 0.05),
        0.05,
        "Reject the unit-root null at 5%; evidence favors stationarity." if test.stat < critical.get("5%", np.inf) else "Do not reject the unit-root null at 5%; evidence favors non-stationarity.",
        critical_value_source="Phillips-Perron critical values",
        specification_label=f"Phillips-Perron trend={trend}",
    )


def difference(y: TimeSeries, order: int = 1, seasonal_period: int | None = None) -> TimeSeries:
    """Apply ordinary and/or seasonal differencing operators."""
    if order < 0:
        raise ValueError("order must be non-negative")
    result = y
    for _ in range(order):
        result = result.diff(1)
    if seasonal_period is not None:
        if seasonal_period < 1 or seasonal_period >= result.nobs:
            raise ValueError("seasonal_period must be positive and smaller than the series length")
        values = result.values[seasonal_period:] - result.values[:-seasonal_period]
        index = result.index[seasonal_period:] if result.index is not None else None
        result = TimeSeries(values, index=index, name=f"DS{seasonal_period}({result.name})", frequency=result.frequency)
    return result


def classify_ts_ds(y: TimeSeries | Iterable[float]) -> dict[str, object]:
    """Classify a series using the sequential DF/ADF workflow."""
    report = dickey_fuller_sequential(y)
    return {
        "tests": report.tests,
        "selected": report.selected,
        "nature": report.nature,
        "is_ts_candidate": "TS candidate" in report.nature,
        "is_ds_candidate": "DS" in report.nature,
    }


def trend_regression(y: TimeSeries | Iterable[float], *, degree: int = 1) -> pd.DataFrame:
    """Estimate a deterministic polynomial trend and return coefficients and residuals."""
    x = _values(y)
    t = np.arange(1, x.size + 1, dtype=float)
    X = np.column_stack([t**j for j in range(degree + 1)])
    beta = np.linalg.lstsq(X, x, rcond=None)[0]
    fitted = X @ beta
    residual = x - fitted
    rows = [{"Term": "Intercept" if j == 0 else f"Trend^{j}", "Coefficient": float(beta[j])} for j in range(degree + 1)]
    return pd.DataFrame(rows).assign(R2=float(1 - np.sum(residual**2) / np.sum((x - x.mean())**2)))
