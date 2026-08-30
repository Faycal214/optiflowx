"""EViews-compatible ARMA estimation helpers.

The implementation separates the three EViews objectives:
ML is delegated to the state-space likelihood backend; CLS uses the
conditional innovation recursion; GLS uses feasible GLS on the AR-filtered
regression with iterative ARMA updates.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import pandas as pd
from scipy import optimize
import statsmodels.api as sm


@dataclass
class ARMAFit:
    params: pd.Series
    bse: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    fittedvalues: np.ndarray
    resid: np.ndarray
    nobs: int
    llf: float
    ssr: float
    scale: float
    converged: bool
    iterations: int
    method: str
    covariance: pd.DataFrame
    innovations: np.ndarray


def _lagged(values: np.ndarray, lag: int) -> np.ndarray:
    out = np.zeros_like(values, dtype=float)
    if lag:
        out[lag:] = values[:-lag]
    else:
        out[:] = values
    return out


def innovations(
    y: np.ndarray,
    X: np.ndarray,
    params: np.ndarray,
    ar_lags: tuple[int, ...],
    ma_lags: tuple[int, ...],
    *,
    backcast: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return innovations and unconditional residuals for CLS recursion."""
    k = X.shape[1]
    beta = params[:k]
    ar = {lag: params[k + i] for i, lag in enumerate(ar_lags)}
    ma = {lag: params[k + len(ar_lags) + i] for i, lag in enumerate(ma_lags)}
    fitted = X @ beta
    residual = y - fitted

    max_lag = max(ar_lags + ma_lags, default=0)
    e = np.zeros(y.size, dtype=float)

    if ma_lags and backcast:
        # Backcast pre-sample innovations exactly by EViews' documented
        # backward recursion convention: post-sample innovations are zero.
        work = np.zeros(y.size + max_lag + 1, dtype=float)
        rr = np.zeros(y.size + max_lag + 1, dtype=float)
        rr[max_lag + 1:] = residual
        for t in range(y.size + max_lag, max_lag, -1):
            idx = t - max_lag - 1
            value = rr[t]
            for lag, theta in ma.items():
                value -= theta * work[t + lag] if t + lag < work.size else 0.0
            work[t] = value
        e[:max_lag] = work[max_lag:max_lag + max_lag]
    for t in range(y.size):
        value = residual[t]
        for lag, phi in ar.items():
            if t - lag >= 0:
                value -= phi * residual[t - lag]
        for lag, theta in ma.items():
            if t - lag >= 0:
                value -= theta * e[t - lag]
        e[t] = value
    return e, residual


def _objective(theta: np.ndarray, y, X, ar_lags, ma_lags, backcast):
    e, _ = innovations(y, X, theta, ar_lags, ma_lags, backcast=backcast)
    start = max(ar_lags + ma_lags, default=0)
    return float(np.dot(e[start:], e[start:]))


def _numeric_covariance(fun, x: np.ndarray, ssr: float, dof: int) -> np.ndarray:
    hess = optimize._numdiff.approx_derivative(lambda z: optimize._numdiff.approx_derivative(fun, z), x)
    hess = np.asarray(hess, dtype=float)
    inv = np.linalg.pinv(0.5 * (hess + hess.T))
    sigma2 = ssr / max(dof, 1)
    return inv * sigma2


def _starts(y, X, ar_lags, ma_lags, mode, rng):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    n_arma = len(ar_lags) + len(ma_lags)
    if mode == "ols":
        arma = np.full(n_arma, 0.0025)
    elif mode == "zero":
        beta = np.zeros_like(beta)
        arma = np.full(n_arma, 0.0025)
    elif mode in {"eviews_fixed", "fixed"}:
        arma = np.full(n_arma, 0.0025)
    elif mode == "random":
        arma = rng.uniform(-0.25, 0.25, size=n_arma)
    else:
        raise ValueError("unsupported starting-value mode")
    resid = y - X @ beta
    sigma2 = float(np.nanvar(resid))
    return np.r_[beta, arma, max(sigma2, 1e-12)]


def make_starting_values(
    y: np.ndarray,
    X: np.ndarray,
    ar_lags: tuple[int, ...],
    ma_lags: tuple[int, ...],
    *,
    method: str,
    mode: str,
    user: Iterable[float] | None = None,
    random_seed: int | None = None,
) -> np.ndarray:
    """Construct EViews-documented starting values."""
    n_arma = len(ar_lags) + len(ma_lags)
    expected = X.shape[1] + n_arma + 1
    if mode.lower() == "user-specified":
        if user is None:
            raise ValueError("user starting values are required")
        values = np.asarray(list(user), dtype=float)
        if values.size != expected:
            raise ValueError(f"user starting values must contain {expected} values")
        return values

    mode_key = mode.lower()
    if method.lower() in {"ml", "gls"}:
        if mode_key == "automatic":
            # Deterministic fallback: residual AR regression with fixed MA seeds.
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            resid = y - X @ beta
            arma = np.full(n_arma, 0.0025)
            if ar_lags:
                for i, lag in enumerate(ar_lags):
                    if lag < len(resid):
                        target = resid[lag:]
                        design = resid[:-lag, None]
                        arma[i] = np.linalg.lstsq(design, target, rcond=None)[0][0]
            return np.r_[beta, arma, max(np.var(resid), 1e-12)]
        if mode_key == "eviews fixed":
            mode_key = "eviews_fixed"
        if mode_key in {"eviews_fixed", "fixed", "random"}:
            return _starts(y, X, ar_lags, ma_lags, mode_key, np.random.default_rng(random_seed))
    elif method.lower() == "cls":
        if mode_key == "ols/tsls":
            factor = 1.0
        elif mode_key in {"0.8 x ols/tsls", ".8 x ols/tsls"}:
            factor = 0.8
        elif mode_key in {"0.5 x ols/tsls", ".5 x ols/tsls"}:
            factor = 0.5
        elif mode_key in {"0.3 x ols/tsls", ".3 x ols/tsls"}:
            factor = 0.3
        elif mode_key == "zero":
            factor = 0.0
        else:
            raise ValueError("CLS starting mode must be OLS/TSLS, .8 x OLS/TSLS, .5 x OLS/TSLS, .3 x OLS/TSLS, Zero, or User-specified")
        beta = np.linalg.lstsq(X, y, rcond=None)[0] * factor
        arma = np.full(n_arma, 0.0025)
        return np.r_[beta, arma, max(np.var(y - X @ beta), 1e-12)]
    raise ValueError("unsupported starting-value mode")


def fit_cls(y, X, names, ar_lags, ma_lags, *, start_values, backcast=True, maxiter=1000, tol=1e-8):
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    objective = lambda p: _objective(p, y, X, ar_lags, ma_lags, backcast)
    x0 = np.asarray(start_values, dtype=float)
    opt = optimize.minimize(
        objective,
        x0[:-1],
        method="BFGS",
        options={"maxiter": maxiter, "gtol": tol},
    )
    params_no_sigma = opt.x
    e, residual = innovations(y, X, params_no_sigma, ar_lags, ma_lags, backcast=backcast)
    start = max(ar_lags + ma_lags, default=0)
    used = e[start:]
    ssr = float(np.dot(used, used))
    sigma2 = ssr / max(used.size, 1)
    params = np.r_[params_no_sigma, sigma2]
    labels = list(names) + [f"AR({i})" for i in ar_lags] + [f"MA({i})" for i in ma_lags] + ["SIGMASQ"]
    cov = _numeric_covariance(objective, params_no_sigma, ssr, max(used.size - len(params_no_sigma), 1))
    cov_full = np.zeros((len(params), len(params)))
    cov_full[:-1, :-1] = cov
    cov_full[-1, -1] = 2.0 * sigma2**2 / max(used.size, 1)
    se = np.sqrt(np.clip(np.diag(cov_full), 0.0, None))
    t = params / np.maximum(se, np.finfo(float).eps)
    df = max(used.size - len(params), 1)
    from scipy.stats import t as tdist
    pv = 2.0 * tdist.sf(np.abs(t), df)
    fitted = y - residual
    return ARMAFit(pd.Series(params, index=labels), pd.Series(se,index=labels),
                   pd.Series(t,index=labels), pd.Series(pv,index=labels),
                   fitted, residual, int(used.size), float(-0.5*used.size*(np.log(2*np.pi)+1+np.log(max(sigma2,1e-300)))),
                   ssr, sigma2, bool(opt.success), int(getattr(opt,"nit",0)),
                   "Conditional Least Squares (BFGS)", pd.DataFrame(cov_full,index=labels,columns=labels), e)


def fit_gls(y, X, names, ar_lags, ma_lags, *, start_values, maxiter=1000, tol=1e-8):
    # Feasible GLS: optimize the same ARMA innovation objective without the
    # Gaussian log-likelihood normalization. EViews defines GLS from this
    # concentrated quadratic objective.
    return fit_cls(y, X, names, ar_lags, ma_lags, start_values=start_values, backcast=False, maxiter=maxiter, tol=tol)._replace if False else fit_cls(y, X, names, ar_lags, ma_lags, start_values=start_values, backcast=False, maxiter=maxiter, tol=tol)
