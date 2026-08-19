"""Stationarity, unit-root, TS/DS and differencing diagnostics.

The DF/ADF implementation follows the USTHB course workflow: select a common
ADF lag order by residual whitening and parsimony, test Model 3 -> Model 2
-> Model 1, validate deterministic terms conditionally, and use explicit
Dickey-Fuller critical-value tables for the unit-root and joint F decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss

from .series import TimeSeries


DFRegression = Literal["n", "c", "ct"]
Decision = Literal["reject", "fail_to_reject"]

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

# USTHB / Dickey-Fuller course-table critical values for the unit-root
# statistic. The course uses tabulated rows n=50, 100, 250 and infinity.
# The 91-observation example explicitly uses the n=100 row.
DF_CRITICAL_VALUES: dict[str, dict[int | str, dict[str, float]]] = {
    "n": {
        50: {"1%": -2.62, "5%": -1.95, "10%": -1.61},
        100: {"1%": -2.60, "5%": -1.95, "10%": -1.61},
        250: {"1%": -2.58, "5%": -1.95, "10%": -1.62},
        "inf": {"1%": -2.58, "5%": -1.95, "10%": -1.62},
    },
    "c": {
        50: {"1%": -3.58, "5%": -2.93, "10%": -2.60},
        100: {"1%": -3.51, "5%": -2.89, "10%": -2.58},
        250: {"1%": -3.46, "5%": -2.88, "10%": -2.57},
        "inf": {"1%": -3.43, "5%": -2.86, "10%": -2.57},
    },
    "ct": {
        50: {"1%": -4.15, "5%": -3.50, "10%": -3.18},
        100: {"1%": -4.04, "5%": -3.45, "10%": -3.15},
        250: {"1%": -3.99, "5%": -3.43, "10%": -3.13},
        "inf": {"1%": -3.96, "5%": -3.41, "10%": -3.12},
    },
}

# Values reproduced from the USTHB DF critical-value tables for F2/F3.
DF_F_CRITICAL_VALUES: dict[str, dict[int | str, dict[str, float]]] = {
    "F2": {
        50: {"10%": 3.94, "5%": 4.86, "1%": 7.06},
        100: {"10%": 3.86, "5%": 4.71, "1%": 6.70},
        250: {"10%": 3.81, "5%": 4.63, "1%": 6.52},
        "inf": {"10%": 3.78, "5%": 4.59, "1%": 6.43},
    },
    "F3": {
        50: {"10%": 5.61, "5%": 6.73, "1%": 9.31},
        100: {"10%": 5.47, "5%": 6.49, "1%": 8.73},
        250: {"10%": 5.39, "5%": 6.34, "1%": 8.43},
        "inf": {"10%": 5.34, "5%": 6.25, "1%": 8.27},
    },
}

COURSE_TABLE_SIZES = (50, 100, 250)


@dataclass(frozen=True)
class SpecificationTestResult:
    """Standard or Dickey-Fuller-specific test used to validate deterministic terms."""

    name: str
    null_hypothesis: str
    alternative_hypothesis: str
    statistic: float
    critical_value: float
    decision: Decision
    alpha: float
    source: str
    detail: str = ""

    @property
    def rejects_null(self) -> bool:
        return self.decision == "reject"

    def summary(self) -> str:
        return (
            f"{self.name}: statistic={self.statistic:.6f}, critical={self.critical_value:.6f}, "
            f"decision={'Reject H0' if self.rejects_null else 'Do not reject H0'}; {self.detail}"
        )


@dataclass(frozen=True)
class UnitRootResult:
    """Unified EViews-style unit-root result."""

    test: str
    statistic: float
    pvalue: float | None
    critical_values: dict[str, float]
    regression: DFRegression
    lags: int
    nobs: int
    null_hypothesis: str
    alternative_hypothesis: str
    decision: Decision
    alpha: float
    conclusion: str
    critical_value_source: str = "USTHB / Dickey-Fuller course critical-value table"
    specification_label: str = ""
    coefficient: float | None = None
    coefficient_name: str = "γ"
    coefficient_tvalue: float | None = None
    coefficient_pvalue: float | None = None

    @property
    def rejects_null(self) -> bool:
        return self.decision == "reject"

    @property
    def decision_rule(self) -> str:
        if self.test == "KPSS Test":
            return f"Reject H0 when test statistic > {self.critical_value:.6f}."
        return f"Reject H0 when test statistic < {self.critical_value:.6f}."

    @property
    def _level_key(self) -> str:
        return {0.01: "1%", 0.05: "5%", 0.10: "10%"}[self.alpha]

    @property
    def critical_value(self) -> float:
        return float(self.critical_values[self._level_key])

    def table(self) -> pd.DataFrame:
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
            f"Coefficient: {self.coefficient_name} = {self.coefficient:.6f}" if self.coefficient is not None else "",
            f"Coefficient t-statistic: {self.coefficient_tvalue:.6f}" if self.coefficient_tvalue is not None else "",
            "",
            f"Test statistic: {self.statistic:.6f}",
        ]
        if self.pvalue is not None and np.isfinite(self.pvalue):
            lines.append(f"Prob.*: {self.pvalue:.6f}  [informational; not the decision rule]")
        lines.extend(
            [
                "",
                "Critical values:",
                *[f"{level}: {value:.6f}" for level, value in self.critical_values.items()],
                "",
                f"Decision level: {self._level_key}",
                f"Decision rule: {self.decision_rule}",
                f"Critical-value source: {self.critical_value_source}",
                f"Decision: {'Reject H0' if self.rejects_null else 'Do not reject H0'}",
                f"Conclusion: {self.conclusion}",
            ]
        )
        return "\n".join(line for line in lines if line != "")

    def interpret(self) -> str:
        return self.conclusion


@dataclass(frozen=True)
class SequentialDFResult:
    """Complete USTHB Model 3 -> Model 2 -> Model 1 DF/ADF decision tree."""

    tests: tuple[UnitRootResult, ...]
    selected: UnitRootResult
    nature: str
    selection_rule: str
    lag_order: int
    lag_selection_method: str
    specification_tests: tuple[SpecificationTestResult, ...] = ()

    @property
    def rejected_at_selected_specification(self) -> bool:
        return self.selected.rejects_null

    def table(self) -> pd.DataFrame:
        rows = []
        spec_map = {item.name: item for item in self.specification_tests}
        for result in self.tests:
            row = {
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
            if result.regression == "ct" and "Model 3 trend test" in spec_map:
                spec = spec_map["Model 3 trend test"]
                row["Deterministic test"] = spec.name
                row["Deterministic statistic"] = spec.statistic
                row["Deterministic critical"] = spec.critical_value
                row["Deterministic decision"] = "Reject H0" if spec.rejects_null else "Do not reject H0"
            elif result.regression == "c" and "Model 2 constant test" in spec_map:
                spec = spec_map["Model 2 constant test"]
                row["Deterministic test"] = spec.name
                row["Deterministic statistic"] = spec.statistic
                row["Deterministic critical"] = spec.critical_value
                row["Deterministic decision"] = "Reject H0" if spec.rejects_null else "Do not reject H0"
            rows.append(row)
        return pd.DataFrame(rows)

    def specification_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Test": item.name,
                    "Null": item.null_hypothesis,
                    "Alternative": item.alternative_hypothesis,
                    "Statistic": item.statistic,
                    "Critical Value": item.critical_value,
                    "Decision": "Reject H0" if item.rejects_null else "Do not reject H0",
                    "Source": item.source,
                }
                for item in self.specification_tests
            ]
        )

    def summary(self) -> str:
        sections = [
            "Dickey-Fuller / Augmented Dickey-Fuller Sequential Test",
            "=" * 76,
            "Course workflow: Model 3 → Model 2 → Model 1",
            f"Common lag order: p = {self.lag_order} ({self.lag_selection_method})",
            "",
            self.table().to_string(index=False, float_format=lambda value: f"{value:.6f}"),
        ]
        if self.specification_tests:
            sections.extend(["", "Conditional specification tests:", self.specification_table().to_string(index=False)])
        sections.extend(
            [
                "",
                f"Selected specification: {self.selected.specification_label}",
                f"Selection rule: {self.selection_rule}",
                f"Series classification: {self.nature}",
                f"Interpretation: {self.interpret()}",
            ]
        )
        return "\n".join(sections)

    def interpret(self) -> str:
        if self.selected.rejects_null:
            return (
                f"The unit-root null is rejected at the selected terminal specification ({self.selected.specification_label}). "
                f"{self.nature}. {self.selection_rule}"
            )
        return (
            f"The unit-root null is not rejected at the terminal specification ({self.selected.specification_label}). "
            f"The series remains a difference-stationary / integrated candidate. {self.selection_rule}"
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
        raise ValueError("alpha must be one of the DF levels: 0.01, 0.05, or 0.10")


def _decision(statistic: float, critical_values: dict[str, float], alpha: float) -> Decision:
    _validate_alpha(alpha)
    level = {0.01: "1%", 0.05: "5%", 0.10: "10%"}[alpha]
    return "reject" if statistic < float(critical_values[level]) else "fail_to_reject"


def _course_table_key(nobs: int) -> tuple[int | str, str]:
    """Use the course's tabulated-size convention: exact row, otherwise next row."""
    if nobs <= 50:
        return 50, "USTHB DF table, n=50"
    if nobs <= 100:
        return 100, "USTHB DF table, n=100"
    if nobs <= 250:
        return 250, "USTHB DF table, n=250"
    return "inf", "USTHB DF table, n=∞"


def _course_df_critical(regression: DFRegression, nobs: int) -> tuple[dict[str, float], str]:
    key, source = _course_table_key(nobs)
    return dict(DF_CRITICAL_VALUES[regression][key]), source


def _course_f_critical(model_name: Literal["F2", "F3"], nobs: int, alpha: float) -> tuple[float, str]:
    _validate_alpha(alpha)
    level = {0.01: "1%", 0.05: "5%", 0.10: "10%"}[alpha]
    key, source = _course_table_key(nobs)
    return DF_F_CRITICAL_VALUES[model_name][key][level], source


def _adf_regression_frame(x: np.ndarray, regression: DFRegression, lags: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Construct the OLS regression used by DF/ADF with a fixed lag order."""
    if not isinstance(lags, int) or lags < 0:
        raise ValueError("lags must be a non-negative integer")
    if lags >= x.size - 2:
        raise ValueError("lags leave too few observations for DF/ADF estimation")
    dy = np.diff(x)
    start = lags
    y = dy[start:]
    columns: list[str] = []
    parts: list[np.ndarray] = []
    if regression in {"c", "ct"}:
        parts.append(np.ones_like(y))
        columns.append("const")
    if regression == "ct":
        trend = np.arange(start + 1, len(x), dtype=float)
        parts.append(trend)
        columns.append("trend")
    parts.append(x[start:-1])
    columns.append("gamma")
    for j in range(1, lags + 1):
        parts.append(dy[start - j : -j])
        columns.append(f"diff_lag_{j}")
    X = np.column_stack(parts)
    return y, X, columns


def sm_ols(y: np.ndarray, X: np.ndarray) -> dict[str, object]:
    """Small OLS helper kept local to avoid coupling the DF engine to the regression API."""
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(X.shape[0] - X.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    stderr = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        tvalues = beta / stderr
    pvalues = 2 * stats.t.sf(np.abs(tvalues), dof)
    return {
        "params": beta,
        "stderr": stderr,
        "tvalues": tvalues,
        "pvalues": pvalues,
        "resid": resid,
        "ssr": float(resid @ resid),
        "nobs": int(X.shape[0]),
        "df_resid": int(dof),
    }


def _fit_df_regression(x: np.ndarray, regression: DFRegression, lags: int) -> dict[str, object]:
    y, X, columns = _adf_regression_frame(x, regression, lags)
    model = sm_ols(y, X)
    return {
        "model": model,
        "params": model["params"],
        "stderr": model["stderr"],
        "tvalues": model["tvalues"],
        "pvalues": model["pvalues"],
        "columns": columns,
        "nobs": len(y),
    }


def _residual_whitening(x: np.ndarray, *, regression: DFRegression, lags: int, test_lags: int, alpha: float) -> tuple[bool, float, int]:
    """Check that ADF residuals are jointly free of autocorrelation through test_lags."""
    fitted = _fit_df_regression(x, regression, lags)
    residuals = np.asarray(fitted["model"]["resid"], dtype=float)
    max_lag = min(int(test_lags), max(1, residuals.size // 5))
    lb = acorr_ljungbox(residuals, lags=list(range(1, max_lag + 1)), return_df=True)
    min_pvalue = float(lb["lb_pvalue"].min())
    return bool(min_pvalue > alpha), min_pvalue, max_lag


def _select_whitened_lag(
    x: np.ndarray,
    *,
    max_lags: int,
    test_lags: int,
    alpha: float,
) -> tuple[int, str]:
    """Select the smallest p that whitens Model 3 residuals, by parsimony."""
    if not isinstance(max_lags, int) or max_lags < 0:
        raise ValueError("max_lags must be a non-negative integer")
    for p in range(max_lags + 1):
        whitened, min_pvalue, checked_lag = _residual_whitening(
            x, regression="ct", lags=p, test_lags=test_lags, alpha=alpha
        )
        if whitened:
            return p, f"minimum p={p} with Ljung-Box whitening through lag {checked_lag} (min p-value={min_pvalue:.4f})"
    whitened, min_pvalue, checked_lag = _residual_whitening(
        x, regression="ct", lags=max_lags, test_lags=test_lags, alpha=alpha
    )
    return max_lags, f"fallback p={max_lags}; whitening not achieved through lag {checked_lag} (min p-value={min_pvalue:.4f})"


def _selected_common_lag(
    x: np.ndarray,
    *,
    max_lags: int | None,
    autolag: str | None,
    alpha: float = 0.05,
    whitening_lags: int = 12,
) -> tuple[int, str]:
    """Choose one common ADF lag order before running the three deterministic models."""
    if max_lags is None:
        max_lags = min(12, max(0, x.size // 5))
    if not isinstance(max_lags, int) or max_lags < 0:
        raise ValueError("max_lags must be a non-negative integer or None")
    if not isinstance(whitening_lags, int) or whitening_lags < 1:
        raise ValueError("whitening_lags must be a positive integer")
    # Keep the explicit fixed-p API used in TPs: autolag=None means use the
    # supplied p exactly. Otherwise apply the course's minimum-whitening rule.
    if autolag is None:
        return max_lags, "fixed course lag order"
    return _select_whitened_lag(
        x,
        max_lags=max_lags,
        test_lags=whitening_lags,
        alpha=alpha,
    )


def _joint_f_test(x: np.ndarray, regression: Literal["ct", "c"], lags: int, alpha: float) -> SpecificationTestResult:
    """Compute F3/F2 for the joint unit-root + deterministic-term null."""
    unrestricted = _fit_df_regression(x, regression, lags)
    y_r, X_ur, cols = _adf_regression_frame(x, regression, lags)
    if regression == "ct":
        keep = [i for i, name in enumerate(cols) if name not in {"gamma", "trend"}]
        name = "Model 3 joint F test"
        null = "H3,0: γ = 0 and β = 0"
        alt = "At least one of γ or β is non-zero"
        model_name = "F3"
    else:
        keep = [i for i, col in enumerate(cols) if col not in {"gamma", "const"}]
        name = "Model 2 joint F test"
        null = "H2,0: γ = 0 and α = 0"
        alt = "At least one of γ or α is non-zero"
        model_name = "F2"
    X_r = X_ur[:, keep] if keep else np.empty((len(y_r), 0))
    if X_r.shape[1]:
        beta_r = np.linalg.lstsq(X_r, y_r, rcond=None)[0]
        restricted_ssr = float(np.sum((y_r - X_r @ beta_r) ** 2))
    else:
        restricted_ssr = float(np.sum(y_r ** 2))
    unrestricted_ssr = float(unrestricted["model"]["ssr"])
    q = 2
    df2 = int(unrestricted["model"]["df_resid"])
    statistic = ((restricted_ssr - unrestricted_ssr) / q) / (unrestricted_ssr / df2)
    critical, source = _course_f_critical(model_name, int(unrestricted["model"]["nobs"]), alpha)
    decision: Decision = "reject" if statistic > critical else "fail_to_reject"
    return SpecificationTestResult(
        name=name,
        null_hypothesis=null,
        alternative_hypothesis=alt,
        statistic=float(statistic),
        critical_value=float(critical),
        decision=decision,
        alpha=alpha,
        source=source,
        detail=f"Compare F to the non-standard {model_name} critical value; ordinary Fisher p-values are not used.",
    )


def _deterministic_term_test(df_result: dict[str, object], regression: Literal["ct", "c"], alpha: float) -> SpecificationTestResult:
    params = np.asarray(df_result["params"])
    tvalues = np.asarray(df_result["tvalues"])
    df_resid = int(df_result["model"]["df_resid"])
    critical = float(stats.t.ppf(1 - alpha / 2, df_resid))
    if regression == "ct":
        idx, label, null, alt = 1, "Model 3 trend test", "H0: β = 0", "H1: β ≠ 0"
    else:
        idx, label, null, alt = 0, "Model 2 constant test", "H0: α = 0", "H1: α ≠ 0"
    statistic = float(tvalues[idx])
    decision: Decision = "reject" if abs(statistic) > critical else "fail_to_reject"
    return SpecificationTestResult(
        name=label,
        null_hypothesis=null,
        alternative_hypothesis=alt,
        statistic=statistic,
        critical_value=critical,
        decision=decision,
        alpha=alpha,
        source=f"Standard two-sided Student-t critical value, df={df_resid}",
        detail=f"Estimated coefficient = {params[idx]:.6f}.",
    )


def adf(
    y: TimeSeries | Iterable[float],
    *,
    regression: DFRegression = "ct",
    lags: int | None = None,
    autolag: str | None = "AIC",
    alpha: float = 0.05,
) -> UnitRootResult:
    """Run an ADF test under one of the course's three deterministic models."""
    x = _values(y)
    if regression not in DF_SPECIFICATIONS:
        raise ValueError("regression must be 'n', 'c', or 'ct'")
    if lags is not None and (not isinstance(lags, int) or lags < 0):
        raise ValueError("lags must be a non-negative integer or None")
    _validate_alpha(alpha)
    result = adfuller(x, regression=regression, maxlag=lags, autolag=autolag)
    if len(result) == 5:
        statistic, pvalue, usedlag, nobs, _statsmodels_critical = result
    elif len(result) == 6:
        statistic, pvalue, usedlag, nobs, _statsmodels_critical, _ = result
    else:
        raise RuntimeError(f"unexpected statsmodels ADF result length: {len(result)}")
    critical_values, source = _course_df_critical(regression, int(nobs))
    decision = _decision(float(statistic), critical_values, alpha)
    reg = _fit_df_regression(x, regression, int(usedlag))
    gamma_index = reg["columns"].index("gamma")
    if decision == "reject":
        conclusion = f"Reject H0 at {int(alpha * 100)}%. Evidence favors stationarity under the {DF_SPECIFICATIONS[regression]['label'].lower()} specification."
    else:
        conclusion = f"Do not reject H0 at {int(alpha * 100)}%. The unit-root null remains plausible under the {DF_SPECIFICATIONS[regression]['label'].lower()} specification."
    return UnitRootResult(
        test="Augmented Dickey-Fuller Test",
        statistic=float(statistic),
        pvalue=float(pvalue),
        critical_values=critical_values,
        regression=regression,
        lags=int(usedlag),
        nobs=int(nobs),
        null_hypothesis="γ = 0 (unit root / non-stationarity under the selected deterministic specification).",
        alternative_hypothesis="γ < 0 (stationarity under the selected deterministic specification).",
        decision=decision,
        alpha=alpha,
        conclusion=conclusion,
        critical_value_source=source,
        specification_label=DF_SPECIFICATIONS[regression]["label"],
        coefficient=float(reg["params"][gamma_index]),
        coefficient_tvalue=float(reg["tvalues"][gamma_index]),
        coefficient_pvalue=float(reg["pvalues"][gamma_index]),
    )


def dickey_fuller(y: TimeSeries | Iterable[float], *, regression: DFRegression = "c", alpha: float = 0.05) -> UnitRootResult:
    """Run the original Dickey-Fuller test with zero augmented lagged differences."""
    return replace(adf(y, regression=regression, lags=0, autolag=None, alpha=alpha), test="Dickey-Fuller Test")


def dickey_fuller_sequential(
    y: TimeSeries | Iterable[float],
    *,
    max_lags: int | None = None,
    autolag: str | None = "AIC",
    alpha: float = 0.05,
    whitening_lags: int = 12,
) -> SequentialDFResult:
    """Apply the complete course-faithful Model 3 -> Model 2 -> Model 1 workflow."""
    _validate_alpha(alpha)
    x = _values(y)
    common_lag, lag_method = _selected_common_lag(
        x,
        max_lags=max_lags,
        autolag=autolag,
        alpha=alpha,
        whitening_lags=whitening_lags,
    )
    models: list[UnitRootResult] = []
    spec_tests: list[SpecificationTestResult] = []

    model3 = adf(x, regression="ct", lags=common_lag, autolag=None, alpha=alpha)
    models.append(model3)
    model3_df = _fit_df_regression(x, "ct", common_lag)
    if model3.rejects_null:
        trend_test = _deterministic_term_test(model3_df, "ct", alpha)
        spec_tests.append(trend_test)
        if trend_test.rejects_null:
            return SequentialDFResult(tuple(models), model3, "stationary around a deterministic trend (TS)", "Model 3 rejected the unit root and the deterministic trend was retained by the conditional β test.", common_lag, lag_method, tuple(spec_tests))
    else:
        f3 = _joint_f_test(x, "ct", common_lag, alpha)
        spec_tests.append(f3)
        if f3.rejects_null:
            return SequentialDFResult(tuple(models), model3, "I(1) with the Model 3 deterministic structure", "Model 3 did not reject the unit-root null and the joint H3,0 test rejected; the series is integrated under the Model 3 specification.", common_lag, lag_method, tuple(spec_tests))

    model2 = adf(x, regression="c", lags=common_lag, autolag=None, alpha=alpha)
    models.append(model2)
    model2_df = _fit_df_regression(x, "c", common_lag)
    if model2.rejects_null:
        constant_test = _deterministic_term_test(model2_df, "c", alpha)
        spec_tests.append(constant_test)
        if constant_test.rejects_null:
            return SequentialDFResult(tuple(models), model2, "stationary around a constant (TS)", "Model 2 rejected the unit root and the constant was retained by the conditional α test.", common_lag, lag_method, tuple(spec_tests))
    else:
        f2 = _joint_f_test(x, "c", common_lag, alpha)
        spec_tests.append(f2)
        if f2.rejects_null:
            return SequentialDFResult(tuple(models), model2, "I(1) with a constant (DS candidate)", "Model 2 did not reject the unit-root null and the joint H2,0 test rejected; the series is integrated with a constant.", common_lag, lag_method, tuple(spec_tests))

    model1 = adf(x, regression="n", lags=common_lag, autolag=None, alpha=alpha)
    models.append(model1)
    nature = "stationary around zero (TS)" if model1.rejects_null else "difference-stationary / integrated candidate (DS)"
    return SequentialDFResult(tuple(models), model1, nature, "After Models 3 and 2 were not retained, the course completes the root test in Model 1 without constant or trend.", common_lag, lag_method, tuple(spec_tests))


def kpss_test(y: TimeSeries | Iterable[float], *, regression: str = "c", nlags: str | int = "auto", alpha: float = 0.05) -> UnitRootResult:
    """Run KPSS as a complementary stationarity diagnostic."""
    x = _values(y)
    statistic, pvalue, lags, critical = kpss(x, regression=regression, nlags=nlags)
    critical_values = {str(k): float(v) for k, v in critical.items()}
    _validate_alpha(alpha)
    level = {0.01: "1%", 0.05: "5%", 0.10: "10%"}[alpha]
    decision: Decision = "reject" if statistic > critical_values[level] else "fail_to_reject"
    conclusion = f"Reject stationarity at {int(alpha * 100)}%; evidence favors non-stationarity." if decision == "reject" else f"Do not reject stationarity at {int(alpha * 100)}%."
    return UnitRootResult(
        "KPSS Test", float(statistic), float(pvalue), critical_values,
        regression if regression in {"n", "c", "ct"} else "c", int(lags), int(x.size - lags),
        "The series is stationary.", "The series is non-stationary.", decision, alpha, conclusion,
        critical_value_source="KPSS critical values", specification_label=f"KPSS regression={regression}"
    )


def phillips_perron(y: TimeSeries | Iterable[float], *, trend: str = "c", lags: int | None = None) -> UnitRootResult:
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
    decision = _decision(float(test.stat), critical, 0.05)
    return UnitRootResult(
        "Phillips-Perron Test", float(test.stat), float(test.pvalue), critical, trend,
        int(test.lags), int(test.nobs), "The series contains a unit root.", "The series is stationary.",
        decision, 0.05,
        "Reject the unit-root null at 5%; evidence favors stationarity." if decision == "reject" else "Do not reject the unit-root null at 5%; evidence favors non-stationarity.",
        critical_value_source="Phillips-Perron critical values", specification_label=f"Phillips-Perron regression={trend}"
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
    """Classify a series with the course's sequential DF/ADF strategy."""
    report = dickey_fuller_sequential(y)
    return {"tests": report.tests, "selected": report.selected, "nature": report.nature, "is_ts_candidate": "TS" in report.nature, "is_ds_candidate": "DS" in report.nature}


def trend_regression(y: TimeSeries | Iterable[float], *, degree: int = 1) -> pd.DataFrame:
    """Estimate a deterministic polynomial trend and return coefficients and residuals."""
    x = _values(y)
    t = np.arange(1, x.size + 1, dtype=float)
    X = np.column_stack([t**j for j in range(degree + 1)])
    beta = np.linalg.lstsq(X, x, rcond=None)[0]
    fitted = X @ beta
    residual = x - fitted
    rows = [{"Term": "Intercept" if j == 0 else f"Trend^{j}", "Coefficient": float(beta[j])} for j in range(degree + 1)]
    denom = np.sum((x - x.mean()) ** 2)
    r2 = float(1 - np.sum(residual**2) / denom) if denom > 0 else float("nan")
    return pd.DataFrame(rows).assign(R2=r2)
