"""Residual validation and specification diagnostics for StochX time-series models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, acorr_ljungbox, het_breuschpagan, het_arch

from .correlogram import CorrelogramResult, correlogram


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
        return bool(self.pvalue < self.alpha) if np.isfinite(self.pvalue) else False

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
    return TestResult("Durbin-Watson", dw, np.nan, "No first-order residual autocorrelation", "Residual autocorrelation is present")


def breusch_godfrey(result, lags: int = 1, *, alpha: float = 0.05) -> TestResult:
    """Run the Breusch-Godfrey LM test on a fitted regression result.

    H0 is no serial correlation through the requested residual lag order.
    The reported statistic is the LM (chi-square) form, matching the
    EViews residual-diagnostics view; the auxiliary-regression F form can be
    obtained from the returned result by calling ``breusch_godfrey_raw``.
    """
    if not isinstance(lags, int) or lags < 1:
        raise ValueError("lags must be a positive integer")
    lm_stat, lm_pvalue, _, _ = acorr_breusch_godfrey(result, nlags=lags, store=False)
    return TestResult(
        "Breusch-Godfrey LM",
        float(lm_stat),
        float(lm_pvalue),
        f"No residual serial correlation through lag {lags}",
        f"Residual serial correlation exists through lag {lags}",
        alpha,
    )


def breusch_godfrey_raw(result, lags: int = 1) -> dict[str, float]:
    """Return both LM and F forms of the Breusch-Godfrey test."""
    if not isinstance(lags, int) or lags < 1:
        raise ValueError("lags must be a positive integer")
    lm_stat, lm_pvalue, f_stat, f_pvalue = acorr_breusch_godfrey(result, nlags=lags, store=False)
    return {
        "LM statistic": float(lm_stat),
        "LM p-value": float(lm_pvalue),
        "F-statistic": float(f_stat),
        "F p-value": float(f_pvalue),
    }


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


def residual_correlogram(residuals, *, lags: int = 12, model_df: int = 0, alpha: float = 0.05) -> CorrelogramResult:
    """Return the frozen EViews-style correlogram for model residuals.

    This is the canonical residual-correlogram entry point for the model
    diagnostics workflow. It delegates all numerical work to
    :func:`stochx.timeseries.correlogram`, so AC, PAC, Q-Stat, Prob., DF,
    confidence bands, missing-value handling, aliases and immutability all
    remain governed by the Stage 8 contract.
    """
    if not isinstance(lags, int) or isinstance(lags, bool) or lags < 1:
        raise ValueError("lags must be a positive integer")
    return correlogram(residuals, nlags=lags, model_df=model_df, alpha=alpha)


def residual_diagnostics_correlogram(result, *, lags: int = 12, alpha: float = 0.05) -> CorrelogramResult:
    """Build the canonical residual correlogram from a fitted TSResult-like model.

    ``model_df`` follows the frozen Stage 8 residual convention: estimated
    AR and MA orders are deducted from the Ljung-Box degrees of freedom.
    """
    order = getattr(result, "order", None)
    p = int(order[0]) if order else 0
    q = int(order[2]) if order and len(order) >= 3 else 0
    return residual_correlogram(result.residuals, lags=lags, model_df=p + q, alpha=alpha)


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



def residual_correlogram_squared(residuals, *, lags: int = 12, model_df: int = 0, alpha: float = 0.05) -> CorrelogramResult:
    """EViews Residual Diagnostics / Correlogram Squared Residuals."""
    x = _clean(residuals)
    if lags < 1 or lags <= model_df:
        raise ValueError("lags must be positive and exceed model_df")
    return correlogram(x ** 2, nlags=lags, model_df=model_df, alpha=alpha)


def histogram_normality(residuals, *, alpha: float = 0.05) -> dict[str, object]:
    """EViews Histogram-Normality diagnostic statistics."""
    x = _clean(residuals)
    jb, pvalue = stats.jarque_bera(x)
    return {
        "Observations": int(x.size),
        "Mean": float(np.mean(x)),
        "Median": float(np.median(x)),
        "Std. Dev.": float(np.std(x, ddof=1)),
        "Skewness": float(stats.skew(x, bias=True)),
        "Kurtosis": float(stats.kurtosis(x, fisher=False, bias=True)),
        "Jarque-Bera": float(jb),
        "Probability": float(pvalue),
        "alpha": float(alpha),
        "Reject normality": bool(pvalue < alpha),
    }


def serial_correlation_lm(
    resid,
    exog,
    lags: int = 1,
    *,
    alpha: float = 0.05,
    model_df: int = 0,
) -> dict[str, float | int]:
    """EViews-style Breusch-Godfrey serial-correlation LM test.

    The auxiliary regression preserves the full estimation sample. Lagged
    residuals before the first available observation are set to zero, which
    matches the EViews documented Uroot convention.
    """
    if not isinstance(lags, int) or isinstance(lags, bool) or lags < 1:
        raise ValueError("lags must be a positive integer")

    e = np.asarray(resid, dtype=float).reshape(-1)
    X = np.asarray(exog, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] != e.size:
        n = min(e.size, X.shape[0])
        e, X = e[-n:], X[-n:]

    if not np.isfinite(e).all() or not np.isfinite(X).all():
        mask = np.isfinite(e) & np.isfinite(X).all(axis=1)
        e, X = e[mask], X[mask]

    n = e.size
    if n <= X.shape[1] + lags:
        raise ValueError("insufficient observations for Breusch-Godfrey test")

    lagged = []
    for j in range(1, lags + 1):
        lag = np.zeros(n, dtype=float)
        lag[j:] = e[:-j]
        lagged.append(lag)

    Z = np.column_stack([X, *lagged])
    aux = sm.OLS(e, sm.add_constant(Z, has_constant="add")).fit()

    lm = float(n * aux.rsquared)
    df = max(int(lags - model_df), 1)
    pvalue = float(stats.chi2.sf(lm, df))

    # EViews reports the F test for the joint null that the added
    # lagged-residual coefficients are zero. statsmodels' aux.fvalue
    # instead tests all non-constant auxiliary regressors.
    r2 = float(aux.rsquared)
    denominator_df = int(n - X.shape[1] - lags)
    if denominator_df <= 0:
        raise ValueError("insufficient denominator degrees of freedom")
    f_stat = (r2 / lags) / ((1.0 - r2) / denominator_df)
    f_pvalue = float(stats.f.sf(f_stat, lags, denominator_df))

    return {
        "LM statistic": lm,
        "Obs*R-squared": lm,
        "p-value": pvalue,
        "df": int(df),
        "F-statistic": float(f_stat),
        "F p-value": f_pvalue,
        "F df numerator": int(lags),
        "F df denominator": denominator_df,
        "nobs": int(n),
    }


def heteroskedasticity_test(residuals, exog, *, test: str = "BPG", lags: int = 12, cross_terms: bool = False, alpha: float = 0.05) -> dict[str, float | int | str]:
    """EViews equation heteroskedasticity test family."""
    kind = test.upper()
    e = _clean(residuals)
    X = np.asarray(exog, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    X = X[np.isfinite(X).all(axis=1)]
    n = min(e.size, X.shape[0])
    e, X = e[-n:], X[-n:]
    if kind in {"BPG", "BREUSCH-PAGAN", "BREUSCH-PAGAN-GODFREY"}:
        lm, pvalue, fstat, fp = het_breuschpagan(e, sm.add_constant(X, has_constant="add"))
        return {"Test": "Breusch-Pagan-Godfrey", "LM statistic": float(lm), "p-value": float(pvalue), "F-statistic": float(fstat), "F p-value": float(fp), "df": int(X.shape[1])}
    if kind == "GLEJSER":
        aux = sm.OLS(np.abs(e), sm.add_constant(X, has_constant="add")).fit()
        lm = float(n * aux.rsquared)
        return {"Test": "Glejser", "LM statistic": lm, "p-value": float(stats.chi2.sf(lm, X.shape[1])), "F-statistic": float(aux.fvalue), "F p-value": float(aux.f_pvalue), "df": int(X.shape[1])}
    if kind == "HARVEY":
        y = np.log(np.maximum(e ** 2, np.finfo(float).tiny))
        aux = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit()
        lm = float(n * aux.rsquared)
        return {"Test": "Harvey", "LM statistic": lm, "p-value": float(stats.chi2.sf(lm, X.shape[1])), "F-statistic": float(aux.fvalue), "F p-value": float(aux.f_pvalue), "df": int(X.shape[1])}
    if kind == "ARCH":
        if not isinstance(lags, int) or lags < 1:
            raise ValueError("lags must be a positive integer")
        y2 = e[lags:] ** 2
        Z = np.column_stack([e[lags - j:-j] ** 2 for j in range(1, lags + 1)])
        aux = sm.OLS(y2, sm.add_constant(Z, has_constant="add")).fit()
        lm = float(y2.size * aux.rsquared)
        return {"Test": "ARCH LM", "LM statistic": lm, "Obs*R-squared": lm, "p-value": float(stats.chi2.sf(lm, lags)), "F-statistic": float(aux.fvalue), "F p-value": float(aux.f_pvalue), "df": int(lags)}
    if kind == "WHITE":
        Z = [X, X ** 2]
        if cross_terms and X.shape[1] > 1:
            crosses = []
            for i in range(X.shape[1]):
                for j in range(i + 1, X.shape[1]):
                    crosses.append((X[:, i] * X[:, j])[:, None])
            if crosses:
                Z.append(np.hstack(crosses))
        design = np.column_stack(Z)
        aux = sm.OLS(e ** 2, sm.add_constant(design, has_constant="add")).fit()
        lm = float(n * aux.rsquared)
        df = max(int(aux.df_model), 1)
        return {"Test": "White", "LM statistic": lm, "Obs*R-squared": lm, "p-value": float(stats.chi2.sf(lm, df)), "F-statistic": float(aux.fvalue), "F p-value": float(aux.f_pvalue), "df": df}
    raise ValueError("test must be BPG, Harvey, Glejser, ARCH, or White")


def chow_breakpoint(y, X, breakpoint: int, *, alpha: float = 0.05) -> dict[str, float | int]:
    """EViews Chow breakpoint test for an OLS equation."""
    y = np.asarray(y, dtype=float).reshape(-1)
    X = sm.add_constant(np.asarray(X, dtype=float), has_constant="add")
    n, k = X.shape
    b = int(breakpoint)
    if b <= k or n - b <= k:
        raise ValueError("each Chow subsample must contain more observations than coefficients")
    pooled = sm.OLS(y, X).fit()
    left = sm.OLS(y[:b], X[:b]).fit()
    right = sm.OLS(y[b:], X[b:]).fit()
    ur_ssr = float(left.ssr + right.ssr)
    r_ssr = float(pooled.ssr)
    q = 2
    fstat = ((r_ssr - ur_ssr) / k) / (ur_ssr / (n - 2 * k))
    pvalue = float(stats.f.sf(fstat, k, n - 2 * k))
    return {"F-statistic": float(fstat), "F p-value": pvalue, "LR statistic": float(n * np.log(max(r_ssr, np.finfo(float).tiny) / max(ur_ssr, np.finfo(float).tiny))), "LR df": int(k), "breakpoint": b}


def recursive_ols_diagnostics(y, X) -> dict[str, np.ndarray]:
    """Compute EViews-style recursive OLS residuals for stability diagnostics."""
    y = np.asarray(y, dtype=float).reshape(-1)
    X = sm.add_constant(np.asarray(X, dtype=float), has_constant="add")
    n, k = X.shape
    if n <= k:
        raise ValueError("insufficient observations for recursive OLS")
    rr = np.full(n, np.nan)
    se = np.full(n, np.nan)
    for t in range(k, n):
        fit = sm.OLS(y[:t], X[:t]).fit()
        x_next = X[t]
        h = float(x_next @ np.linalg.pinv(X[:t].T @ X[:t]) @ x_next)
        variance = float(fit.mse_resid)
        denom = np.sqrt(max(variance * (1.0 + h), np.finfo(float).tiny))
        rr[t] = float((y[t] - x_next @ fit.params) / denom)
        se[t] = 1.0
    return {"recursive_residuals": rr, "recursive_se": se}


def cusum_tests(y, X, *, alpha: float = 0.05) -> dict[str, object]:
    """Return EViews-style recursive residuals, CUSUM and CUSUMSQ series."""
    rec = recursive_ols_diagnostics(y, X)
    rr = rec["recursive_residuals"]
    valid = rr[np.isfinite(rr)]
    if valid.size < 3:
        raise ValueError("insufficient recursive residuals for CUSUM")
    cusum = np.cumsum(valid) / np.sqrt(np.sum(valid**2))
    cusumsq = np.cumsum(valid**2) / np.sum(valid**2)
    t = np.arange(1, valid.size + 1)
    # EViews' CUSUM plots are assessed against 5% boundary lines. We expose
    # the normalized paths here; exact graphical critical lines are stored
    # separately from the statistical series.
    return {
        "recursive_residuals": rr,
        "recursive_se": rec["recursive_se"],
        "CUSUM": cusum,
        "CUSUMSQ": cusumsq,
        "period": t,
        "alpha": float(alpha),
    }

def chow_forecast(y, X, forecast_start: int, *, alpha: float = 0.05) -> dict[str, float | int]:
    """EViews-style Chow forecast stability test."""
    y = np.asarray(y, dtype=float).reshape(-1)
    X = sm.add_constant(np.asarray(X, dtype=float), has_constant="add")
    n, k = X.shape
    start = int(forecast_start)
    if start <= k or start >= n:
        raise ValueError("forecast_start must leave enough estimation and forecast observations")
    restricted = sm.OLS(y[:start], X[:start]).fit()
    forecast_errors = y[start:] - X[start:] @ restricted.params
    rss_forecast = float(np.sum(forecast_errors ** 2))
    rss_full = float(sm.OLS(y, X).fit().ssr)
    m = n - start
    fstat = ((rss_full - rss_forecast) / max(m, 1)) / (rss_forecast / max(start - k, 1))
    pvalue = float(stats.f.sf(fstat, m, start - k))
    return {
        "F-statistic": float(fstat),
        "F p-value": pvalue,
        "forecast_start": start,
        "forecast_observations": int(m),
        "restricted_SSR": rss_forecast,
        "full_SSR": rss_full,
    }
def recursive_coefficient_estimates(y, X) -> dict[str, np.ndarray]:
    """Compute coefficient paths and recursive standard errors."""
    y = np.asarray(y, dtype=float).reshape(-1)
    X = sm.add_constant(np.asarray(X, dtype=float), has_constant="add")
    n, k = X.shape
    if n <= k:
        raise ValueError("insufficient observations for recursive estimates")
    estimates = np.full((n, k), np.nan)
    ses = np.full((n, k), np.nan)
    for t in range(k, n + 1):
        fit = sm.OLS(y[:t], X[:t]).fit()
        estimates[t - 1] = fit.params
        if t > k:
            ses[t - 1] = fit.bse
    return {"coefficients": estimates, "standard_errors": ses, "period": np.arange(n)}
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
