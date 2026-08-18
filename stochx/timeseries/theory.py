"""Theoretical ARMA/Wold calculations used alongside empirical analysis."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def polynomial_roots(coefficients: Iterable[float]) -> np.ndarray:
    """Return roots of a lag polynomial written in ascending powers of L."""
    coeff = np.asarray(list(coefficients), dtype=float)
    if coeff.ndim != 1 or coeff.size < 2:
        return np.asarray([], dtype=complex)
    return np.roots(coeff[::-1])


def is_stationary_ar(phi: Iterable[float]) -> bool:
    """Check the course root condition for AR stationarity."""
    phi = np.asarray(list(phi), dtype=float)
    return bool(np.all(np.abs(polynomial_roots([1.0, *(-phi)])) > 1.0))


def is_invertible_ma(theta: Iterable[float]) -> bool:
    """Check the course root condition for MA invertibility."""
    theta = np.asarray(list(theta), dtype=float)
    return bool(np.all(np.abs(polynomial_roots([1.0, *theta])) > 1.0))


def process_mean(phi: Iterable[float], c: float) -> float:
    """Compute the stationary AR/ARMA mean c/(1-sum(phi))."""
    coefficients = np.asarray(list(phi), dtype=float)
    denominator = 1.0 - float(coefficients.sum())
    if np.isclose(denominator, 0.0):
        raise ValueError("stationary mean is undefined when 1-sum(phi)=0")
    return float(c / denominator)


def impulse_response(phi: Iterable[float], theta: Iterable[float], n: int = 20) -> np.ndarray:
    """Compute the first ``n`` MA(infinity) coefficients Ψ(L)=Θ(L)/Φ(L)."""
    if n < 1:
        raise ValueError("n must be positive")
    ar = np.asarray(list(phi), dtype=float)
    ma = np.asarray(list(theta), dtype=float)
    psi = np.zeros(n, dtype=float)
    psi[0] = 1.0
    for k in range(1, n):
        theta_k = ma[k - 1] if k <= ma.size else 0.0
        ar_part = np.dot(ar[: min(k, ar.size)], psi[k - np.arange(1, min(k, ar.size) + 1)])
        psi[k] = theta_k + ar_part
    return psi


def inverse_ar_coefficients(theta: Iterable[float], n: int = 20) -> np.ndarray:
    """Compute the AR(infinity) coefficients of an invertible MA polynomial."""
    theta = np.asarray(list(theta), dtype=float)
    if n < 1:
        raise ValueError("n must be positive")
    if not is_invertible_ma(theta):
        raise ValueError("MA polynomial is not invertible")
    pi = np.zeros(n, dtype=float)
    for k in range(1, n):
        theta_k = theta[k - 1] if k <= theta.size else 0.0
        previous = np.dot(theta[: min(k, theta.size)], pi[k - np.arange(1, min(k, theta.size) + 1)])
        pi[k] = -(theta_k + previous)
    return pi


def theoretical_ma_acf(theta: Iterable[float], max_lag: int | None = None) -> np.ndarray:
    """Return the theoretical ACF of a finite MA(q)."""
    theta = np.asarray(list(theta), dtype=float)
    coefficients = np.r_[1.0, theta]
    q = theta.size
    max_lag = q if max_lag is None else min(max_lag, q)
    autocov = np.array([np.sum(coefficients[: q + 1 - k] * coefficients[k:]) if k <= q else 0.0 for k in range(max_lag + 1)])
    return autocov / autocov[0]


def theoretical_ar_acf(phi: Iterable[float], max_lag: int = 20) -> np.ndarray:
    """Compute the theoretical AR(p) ACF from the Yule-Walker equations."""
    phi = np.asarray(list(phi), dtype=float)
    p = phi.size
    if p == 0 or max_lag < p:
        raise ValueError("phi must be non-empty and max_lag must be at least p")

    A = np.zeros((p, p), dtype=float)
    b = np.zeros(p, dtype=float)
    for k in range(1, p + 1):
        row = k - 1
        for j in range(1, p + 1):
            lag = abs(k - j)
            if lag == 0:
                b[row] += phi[j - 1]
            else:
                A[row, lag - 1] -= phi[j - 1]
        A[row, k - 1] += 1.0

    try:
        rho = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Yule-Walker system is singular") from exc

    result = np.empty(max_lag + 1, dtype=float)
    result[0] = 1.0
    result[1 : p + 1] = rho
    for k in range(p + 1, max_lag + 1):
        result[k] = float(np.dot(phi, result[k - np.arange(1, p + 1)]))
    return result
