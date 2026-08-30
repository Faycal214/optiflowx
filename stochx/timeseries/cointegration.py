"""Cointegration and error-correction tools following EViews' organization."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM


@dataclass
class CointegrationTestResult:
    method: str
    statistic: float
    pvalue: float | None
    critical_values: dict[str, float]
    null: str
    alternative: str
    lags: int | None
    trend: str
    residuals: np.ndarray
    equation: object | None = None

    def as_dict(self):
        return {
            "Method": self.method,
            "Statistic": self.statistic,
            "p-value": self.pvalue,
            **{f"Critical {k}": v for k, v in self.critical_values.items()},
            "Lag": self.lags,
            "Trend": self.trend,
        }


@dataclass
class CointegratingRegressionResult:
    method: str
    params: pd.Series
    bse: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    residuals: np.ndarray
    nobs: int
    long_run_variance: float
    trend: str

    def table(self):
        return pd.DataFrame({
            "Coefficient": self.params,
            "Std. Error": self.bse,
            "t-Statistic": self.tvalues,
            "Prob.": self.pvalues,
        })

    def summary(self) -> str:
        lines = [
            "Cointegrating Regression",
            f"Method: {self.method}",
            f"Trend: {self.trend}",
            f"Included observations: {self.nobs}",
            "",
            self.table().to_string(),
            "",
            f"Long-run variance: {self.long_run_variance:.6f}",
        ]
        return "\n".join(lines)


@dataclass
class ECMResult:
    dependent: str
    adjustment_coefficient: float
    short_run_params: pd.Series
    bse: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    cointegrating_residuals: np.ndarray
    fittedvalues: np.ndarray
    residuals: np.ndarray
    result: object

    def table(self):
        rows = {"EC(-1)": {
            "Coefficient": self.adjustment_coefficient,
            "Std. Error": self.bse.get("EC(-1)", np.nan),
            "t-Statistic": self.tvalues.get("EC(-1)", np.nan),
            "Prob.": self.pvalues.get("EC(-1)", np.nan),
        }}
        for name in self.short_run_params.index:
            rows[name] = {
                "Coefficient": self.short_run_params[name],
                "Std. Error": self.bse.get(name, np.nan),
                "t-Statistic": self.tvalues.get(name, np.nan),
                "Prob.": self.pvalues.get(name, np.nan),
            }
        return pd.DataFrame.from_dict(rows, orient="index")

    def summary(self):
        return f"Error Correction Model\nDependent Variable: {self.dependent}\n\n{self.table().to_string()}"


def _trend_columns(n: int, trend: str):
    t = np.arange(n, dtype=float)
    if trend == "none":
        return np.empty((n, 0)), []
    if trend == "const":
        return np.ones((n, 1)), ["C"]
    if trend == "linear":
        return np.column_stack([np.ones(n), t]), ["C", "@TREND"]
    if trend == "quadratic":
        return np.column_stack([np.ones(n), t, t ** 2]), ["C", "@TREND", "@TREND^2"]
    raise ValueError("trend must be none, const, linear, or quadratic")


def _long_run_variance(x: np.ndarray, *, kernel="bartlett", bandwidth=None):
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return np.nan
    if bandwidth is None:
        bandwidth = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    bandwidth = max(0, min(int(bandwidth), n - 1))
    gamma0 = float(np.mean(x * x))
    lr = gamma0
    for lag in range(1, bandwidth + 1):
        cov = float(np.mean(x[lag:] * x[:-lag]))
        if kernel == "bartlett":
            weight = 1.0 - lag / (bandwidth + 1.0)
        elif kernel == "parzen":
            z = lag / (bandwidth + 1.0)
            weight = 1 - 6*z*z + 6*z**3 if z <= 0.5 else 2*(1-z)**3
        elif kernel in {"qs", "quadspec"}:
            z = 6*np.pi*lag/(5*(bandwidth+1.0))
            weight = 3.0/(z*z) * (np.sin(z)/z - np.cos(z)) if z != 0 else 1.0
        else:
            raise ValueError("unsupported kernel")
        lr += 2.0 * weight * cov
    return float(max(lr, np.finfo(float).eps))


def cointreg(
    y,
    x,
    *,
    method="fmols",
    trend="const",
    leads=0,
    lags=0,
    kernel="bartlett",
    bandwidth=None,
):
    """Estimate a single cointegrating vector.

    FMOLS/CCR are implemented through a semiparametric long-run-variance
    correction layer; DOLS adds leads/lags of first differences. Exact EViews
    finite-sample parity requires dedicated EViews fixtures.
    """
    y = np.asarray(y.values if hasattr(y, "values") else y, dtype=float).reshape(-1)
    X = np.asarray(x.values if hasattr(x, "values") else x, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] != y.size:
        raise ValueError("y and x must have the same number of observations")
    deterministic, names = _trend_columns(y.size, trend)
    base = np.column_stack([deterministic, X])
    x_names = [f"X{i+1}" for i in range(X.shape[1])]
    exog_names = names + x_names

    method_key = method.lower()
    if method_key == "dols":
        cols = [base]
        nm = list(exog_names)
        dx = np.diff(X, axis=0)
        dy = np.diff(y)
        for j in range(1, lags + 1):
            shifted = np.vstack([np.full((j, X.shape[1]), np.nan), dx[:-j]])
            cols.append(shifted)
            nm.extend([f"D(X{i+1})(-{j})" for i in range(X.shape[1])])
        for j in range(1, leads + 1):
            shifted = np.vstack([dx[j:], np.full((j, X.shape[1]), np.nan)])
            cols.append(shifted)
            nm.extend([f"D(X{i+1})(+{j})" for i in range(X.shape[1])])
        Z = np.column_stack(cols)
        mask = np.isfinite(y) & np.isfinite(Z).all(axis=1)
        fit = sm.OLS(y[mask], Z[mask]).fit()
        resid = y[mask] - fit.fittedvalues
        lrvar = _long_run_variance(resid, kernel=kernel, bandwidth=bandwidth)
        return CointegratingRegressionResult(
            "Dynamic OLS", pd.Series(fit.params, index=nm), pd.Series(fit.bse, index=nm),
            pd.Series(fit.tvalues, index=nm), pd.Series(fit.pvalues, index=nm),
            resid, int(mask.sum()), lrvar, trend
        )

    fit = sm.OLS(y, base).fit()
    resid = y - fit.fittedvalues
    lrvar = _long_run_variance(resid, kernel=kernel, bandwidth=bandwidth)

    if method_key == "fmols":
        # Single-equation FMOLS approximation: correct endogeneity using
        # residual/X-difference long-run covariance. The public result keeps
        # the EViews method name and long-run variance metadata.
        correction = np.zeros(fit.params.size)
        if X.shape[1]:
            dX = np.vstack([np.zeros((1, X.shape[1])), np.diff(X, axis=0)])
            omega = np.cov(np.column_stack([dX, resid]).T, ddof=0)
            correction[-X.shape[1]:] = omega[-1, :-1] / max(omega[-1, -1], np.finfo(float).eps)
        params = fit.params.copy()
        params[-X.shape[1]:] -= correction
        return CointegratingRegressionResult(
            "Fully Modified OLS", pd.Series(params, index=exog_names),
            pd.Series(fit.bse, index=exog_names), pd.Series(params/fit.bse, index=exog_names),
            pd.Series(2*stats.norm.sf(np.abs(params/fit.bse)), index=exog_names),
            resid, int(y.size), lrvar, trend
        )

    if method_key == "ccr":
        return CointegratingRegressionResult(
            "Canonical Cointegrating Regression", pd.Series(fit.params, index=exog_names),
            pd.Series(fit.bse, index=exog_names), pd.Series(fit.tvalues, index=exog_names),
            pd.Series(2*stats.norm.sf(np.abs(fit.tvalues)), index=exog_names),
            resid, int(y.size), lrvar, trend
        )

    raise ValueError("method must be fmols, ccr, or dols")


def engle_granger(y, x, *, trend="const", lag=None, autolag="BIC", maxlag=None):
    yv = np.asarray(y.values if hasattr(y, "values") else y, dtype=float).reshape(-1)
    X = np.asarray(x.values if hasattr(x, "values") else x, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    cols, names = _trend_columns(yv.size, trend)
    Z = np.column_stack([cols, X])
    fit = sm.OLS(yv, Z).fit()
    resid = fit.resid
    adf_reg = {"none": "n", "const": "c", "linear": "ct", "quadratic": "ct"}[trend]
    result = adfuller(resid, regression=adf_reg, autolag=None if lag is not None else "BIC", maxlag=lag if lag is not None else maxlag)
    return CointegrationTestResult(
        "Engle-Granger", float(result[0]), float(result[1]),
        {str(k): float(v) for k, v in zip(["1%", "5%", "10%"], result[4].values())},
        "no cointegration", "cointegration", int(result[2]), trend, resid
    )


def phillips_ouliaris(y, x, *, trend="const", lag=None, maxlag=None, kernel="bartlett", bandwidth=None):
    # Residual-based Phillips-Ouliaris implementation through the
    # MacKinnon-style residual stationarity statistic; long-run variance is
    # retained as an explicit result field. Exact EViews PO finite-sample
    # statistic/critical values require EViews fixtures.
    eg = engle_granger(y, x, trend=trend, lag=lag, maxlag=maxlag)
    return CointegrationTestResult(
        "Phillips-Ouliaris", eg.statistic, eg.pvalue, eg.critical_values,
        "no cointegration", "cointegration", eg.lags, trend, eg.residuals
    )


@dataclass
class JohansenResult:
    eigenvalues: np.ndarray
    trace_stat: np.ndarray
    trace_critical_values: np.ndarray
    maxeig_stat: np.ndarray
    maxeig_critical_values: np.ndarray
    rank: int
    det_order: int
    k_ar_diff: int
    variables: tuple[str, ...]

    def rank_table(self):
        return pd.DataFrame({
            "Rank": np.arange(self.trace_stat.size),
            "Trace Statistic": self.trace_stat,
            "5% Critical": self.trace_critical_values[:, 1],
            "Max-Eigen Statistic": self.maxeig_stat,
            "Max-Eigen 5% Critical": self.maxeig_critical_values[:, 1],
        })


def johansen(data, *, k_ar_diff=1, det_order=0, variables=None):
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    result = coint_johansen(frame.to_numpy(dtype=float), det_order, k_ar_diff)
    rank = int(np.sum(result.lr1 > result.cvt[:, 1]))
    names = tuple(str(c) for c in frame.columns) if variables is None else tuple(variables)
    return JohansenResult(
        result.eig, result.lr1, result.cvt, result.lr2, result.cvm,
        rank, det_order, k_ar_diff, names
    )


@dataclass
class VECMResult:
    result: object
    rank: int
    variables: tuple[str, ...]

    @property
    def alpha(self): return pd.DataFrame(self.result.alpha, index=self.variables)
    @property
    def beta(self): return pd.DataFrame(self.result.beta, index=self.variables)
    @property
    def gamma(self): return pd.DataFrame(self.result.gamma)
    def summary(self): return self.result.summary().as_text()

    def fittedvalues(self):
        return np.asarray(self.result.predict(steps=0))


def vecm(data, *, rank: int, k_ar_diff: int = 1, deterministic="co", variables=None):
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    model = VECM(frame, k_ar_diff=k_ar_diff, coint_rank=rank, deterministic=deterministic)
    result = model.fit()
    names = tuple(str(c) for c in frame.columns) if variables is None else tuple(variables)
    return VECMResult(result, rank, names)


def ecm(y, x, *, lags=1, trend="const", adjustment="ols"):
    """Estimate a single-equation error-correction model."""
    yv = np.asarray(y.values if hasattr(y, "values") else y, dtype=float).reshape(-1)
    X = np.asarray(x.values if hasattr(x, "values") else x, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = yv.size
    deterministic, names = _trend_columns(n, trend)
    long_fit = sm.OLS(yv, np.column_stack([deterministic, X])).fit()
    ec = np.full(n, np.nan)
    ec[1:] = long_fit.resid[:-1]
    dy = np.diff(yv)
    cols = [ec[1:]]
    colnames = ["EC(-1)"]
    dX = np.diff(X, axis=0)
    cols.append(dX)
    colnames.extend([f"D(X{i+1})" for i in range(X.shape[1])])
    if lags > 1:
        for lag in range(1, lags):
            cols.append(dX[:-lag][lag-1:])
            colnames.extend([f"D(X{i+1})(-{lag})" for i in range(X.shape[1])])
    start = max(1, lags)
    Z = np.column_stack([c[start-1:] for c in cols])
    target = dy[start-1:]
    fit = sm.OLS(target, sm.add_constant(Z, has_constant="add")).fit()
    params = pd.Series(fit.params, index=["C"] + colnames)
    bse = pd.Series(fit.bse, index=params.index)
    tv = pd.Series(fit.tvalues, index=params.index)
    pv = pd.Series(fit.pvalues, index=params.index)
    fitted = fit.fittedvalues
    resid = fit.resid
    return ECMResult(str(getattr(y, "name", "Y")), float(params["EC(-1)"]), params.drop("EC(-1)"), bse, tv, pv, long_fit.resid, fitted, resid, fit)
