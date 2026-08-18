"""Residual validation and specification diagnostics for StochX time-series models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan, het_arch


@dataclass(frozen=True)
class TestResult:
    """Generic hypothesis-test result with a course-oriented interpretation."""

    name: str
    statistic: float
    pvalue: float
    null_hypothesis: str
    alternative: str
    alpha: float = 0.05

    @property
    def reject(self) -> bool:
        """Whether the null hypothesis is rejected at ``alpha``."""
        return bool(self.pvalue < self.alpha)

    @property
    def conclusion(self) -> str:
        """Return a direct accept/reject interpretation."""
        action = "Reject" if self.reject else "Do not reject"
        return f"{action} H0 at the {self.alpha:.0%} level."

    def __str__(self) -> str:
        return f"{self.name}: statistic={self.statistic:.6g}, p-value={self.pvalue:.6g}. {self.conclusion}"


def durbin_watson_test(residuals) -> TestResult:
    """Return the Durbin-Watson statistic for first-order residual autocorrelation."""
    x = _clean(residuals)
    dw = float(durbin_watson(x))
    # DW is not naturally a single-p-value test, so expose distance from 2 as a descriptive test object.
    return TestResult("Durbin-Watson", dw, np.nan, "No first-order residual autocorrelation", "Residual autocorrelation is present")


def box_pierce(residuals, lags: int = 12, *, model_df: int = 0, alpha: float = 0.05) -> TestResult:
    """Run the Box-Pierce portmanteau test H0: rho_1=...=rho_K=0."""
    x = _clean(residuals)
    if lags <= model_df:
        raise ValueError("lags must exceed model_df")
    ac = _acf(x, lags)
    q = x.size * float(np.sum(ac[1:] ** 2))
    pvalue = float(stats.chi2.sf(q, lags - model_df))
    return TestResult("Box-Pierce", q, pvalue, "All residual autocorrelations through K are zero", "At least one residual autocorrelation is non-zero", alpha)


def ljung_box(residuals, lags: int = 12, *, model_df: int = 0, alpha: float = 0.05) -> TestResult:
    """Run the Ljung-Box portmanteau test H0: no residual autocorrelation through K."""
    x = _clean(residuals)
    if lags <= model_df:
        raise ValueError("lags must exceed model_df")
    result = acorr_ljungbox(x, lags=[lags], model_df=model_df, return_df=True).iloc[-1]
    return TestResult("Ljung-Box", float(result["lb_stat"]), float(result["lb_pvalue"]), "All residual autocorrelations through K are zero", "At least one residual autocorrelation is non-zero", alpha)


def jarque_bera(residuals, *, alpha: float = 0.05) -> TestResult:
    """Run the Jarque-Bera normality test."""
    x = _clean(residuals)
    statistic, pvalue = stats.jarque_bera(x)
    return TestResult("Jarque-Bera", float(statistic), float(pvalue), "Residuals are normally distributed", "Residuals are not normally distributed", alpha)


def mean_zero_test(residuals, *, alpha: float = 0.05) -> TestResult:
    """Test the null hypothesis that residual mean equals zero."""
    x = _clean(residuals)
    statistic, pvalue = stats.ttest_1samp(x, 0.0)
    return TestResult("Residual mean", float(statistic), float(pvalue), "E(e_t)=0", "E(e_t) differs from zero", alpha)


def normality_ks(residuals, *, alpha: float = 0.05) -> TestResult:
    """Run a Kolmogorov-Smirnov normality test after standardization."""
    x = _clean(residuals)
    standardized = (x - x.mean()) / x.std(ddof=1)
    statistic, pvalue = stats.kstest(standardized, "norm")
    return TestResult("Kolmogorov-Smirnov", float(statistic), float(pvalue), "Residuals are normal", "Residuals are non-normal", alpha)


def variance_ratio_test(residuals, split: float = 0.5, *, alpha: float = 0.05) -> TestResult:
    """Compare residual variances over two subperiods using an F test."""
    x = _clean(residuals)
    if not 0 < split < 1:
        raise ValueError("split must lie between 0 and 1")
    k = max(2, int(x.size * split))
    a, b = x[:k], x[k:]
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    ratio = float(va / vb if vb else np.inf)
    pvalue = float(2 * min(stats.f.cdf(ratio, len(a) - 1, len(b) - 1), stats.f.sf(ratio, len(a) - 1, len(b) - 1)))
    return TestResult("Variance ratio", ratio, pvalue, "The two residual variances are equal", "The two residual variances differ", alpha)


def breusch_pagan(residuals, exog, *, alpha: float = 0.05) -> TestResult:
    """Run the Breusch-Pagan heteroskedasticity test."""
    y = _clean(residuals)
    x = np.asarray(exog, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    lm, pvalue, _, _ = het_breuschpagan(y, x)
    return TestResult("Breusch-Pagan", float(lm), float(pvalue), "Residual variance is constant", "Residual variance depends on regressors", alpha)


def arch_test(residuals, lags: int = 12, *, alpha: float = 0.05) -> TestResult:
    """Run Engle's ARCH LM test for conditional heteroskedasticity."""
    x = _clean(residuals)
    lm, pvalue, _, _ = het_arch(x, nlags=lags)
    return TestResult("ARCH LM", float(lm), float(pvalue), "No ARCH effects", "Conditional heteroskedasticity is present", alpha)


def roots_report(ar_roots=None, ma_roots=None) -> dict[str, object]:
    """Assess AR stationarity and MA invertibility from polynomial roots."""
    ar = np.asarray([] if ar_roots is None else ar_roots, dtype=complex)
    ma = np.asarray([] if ma_roots is None else ma_roots, dtype=complex)
    ar_mod = np.abs(ar)
    ma_mod = np.abs(ma)
    return {
        "AR roots": ar,
        "MA roots": ma,
        "AR root moduli": ar_mod,
        "MA root moduli": ma_mod,
        "stationary": bool(np.all(ar_mod > 1.0)) if ar.size else True,
        "invertible": bool(np.all(ma_mod > 1.0)) if ma.size else True,
    }


def redundancy_check(ar_roots, ma_roots, *, tolerance: float = 1e-5) -> dict[str, object]:
    """Check the course's AR/MA common-root redundancy condition."""
    ar = np.asarray(ar_roots, dtype=complex)
    ma = np.asarray(ma_roots, dtype=complex)
    matches = []
    for r in ar:
        if ma.size:
            distances = np.abs(ma - r)
            j = int(np.argmin(distances))
            if distances[j] <= tolerance:
                matches.append((r, ma[j]))
    return {"redundant": bool(matches), "common_roots": matches, "recommend_minimal_model": bool(matches)}


def residual_diagnostics(residuals, *, lags: int = 12, p: int = 0, q: int = 0, alpha: float = 0.05) -> pd.DataFrame:
    """Return the standard Box-Jenkins residual validation battery."""
    tests = [
        mean_zero_test(residuals, alpha=alpha),
        box_pierce(residuals, lags=lags, model_df=p + q, alpha=alpha),
        ljung_box(residuals, lags=lags, model_df=p + q, alpha=alpha),
        jarque_bera(residuals, alpha=alpha),
        normality_ks(residuals, alpha=alpha),
        variance_ratio_test(residuals, alpha=alpha),
        arch_test(residuals, lags=min(lags, max(1, len(_clean(residuals)) // 5)), alpha=alpha),
    ]
    return pd.DataFrame(
        [{"Test": t.name, "Statistic": t.statistic, "p-value": t.pvalue, "Reject H0": t.reject, "Conclusion": t.conclusion} for t in tests]
    )


def _clean(values) -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[~np.isnan(x)]
    if x.size < 3:
        raise ValueError("at least three finite observations are required")
    if np.any(np.isinf(x)):
        raise ValueError("values must be finite")
    return x


def _acf(values: np.ndarray, lags: int) -> np.ndarray:
    x = values - values.mean()
    denom = np.dot(x, x)
    out = np.ones(lags + 1)
    for k in range(1, lags + 1):
        out[k] = np.dot(x[k:], x[:-k]) / denom
    return out
