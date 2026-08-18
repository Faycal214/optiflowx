"""Simulation of white noise, AR, MA, ARMA, random walk, SARMA and SARIMA processes."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .series import TimeSeries


def _rng(rng=None):
    return np.random.default_rng(rng)


def white_noise(n: int, *, mean: float = 0.0, scale: float = 1.0, distribution: str = "normal", rng=None) -> TimeSeries:
    """Simulate Gaussian or Student-t white noise."""
    if n < 1 or scale <= 0:
        raise ValueError("n must be positive and scale must be positive")
    r = _rng(rng)
    if distribution == "normal":
        values = r.normal(mean, scale, n)
    elif distribution in {"t", "student"}:
        values = mean + scale * r.standard_t(df=5, size=n) / np.sqrt(5 / 3)
    else:
        raise ValueError("distribution must be 'normal' or 't'")
    return TimeSeries(values, name="WN")


def ar(p: int, phi: Iterable[float], n: int, *, c: float = 0.0, sigma: float = 1.0, burnin: int = 500, rng=None) -> TimeSeries:
    """Simulate an AR(p) process using its lag representation."""
    phi = np.asarray(list(phi), dtype=float)
    if p != phi.size or p < 1:
        raise ValueError("p must equal len(phi) and be positive")
    if sigma <= 0 or n < 1:
        raise ValueError("sigma and n must be positive")
    r = _rng(rng)
    total = n + max(burnin, p + 10)
    eps = r.normal(0.0, sigma, total)
    x = np.zeros(total)
    for t in range(p, total):
        x[t] = c + np.dot(phi, x[t - np.arange(1, p + 1)]) + eps[t]
    return TimeSeries(x[-n:], name=f"AR({p})")


def ma(q: int, theta: Iterable[float], n: int, *, mean: float = 0.0, sigma: float = 1.0, burnin: int = 100, rng=None) -> TimeSeries:
    """Simulate an MA(q) process."""
    theta = np.asarray(list(theta), dtype=float)
    if q != theta.size or q < 1:
        raise ValueError("q must equal len(theta) and be positive")
    r = _rng(rng)
    total = n + max(burnin, q + 5)
    eps = r.normal(0.0, sigma, total + q)
    x = np.full(total, mean, dtype=float)
    for t in range(total):
        x[t] += eps[t + q] + np.dot(theta, eps[t + q - np.arange(1, q + 1)])
    return TimeSeries(x[-n:], name=f"MA({q})")


def arma(p: int, q: int, phi: Iterable[float], theta: Iterable[float], n: int, *, c: float = 0.0, sigma: float = 1.0, burnin: int = 500, rng=None) -> TimeSeries:
    """Simulate an ARMA(p,q) process."""
    phi = np.asarray(list(phi), dtype=float)
    theta = np.asarray(list(theta), dtype=float)
    if phi.size != p or theta.size != q:
        raise ValueError("coefficient lengths must match p and q")
    total = n + max(burnin, p + q + 20)
    r = _rng(rng)
    eps = r.normal(0, sigma, total + q)
    x = np.zeros(total)
    for t in range(max(p, 1), total):
        ar_part = np.dot(phi, x[t - np.arange(1, p + 1)]) if p else 0.0
        ma_part = eps[t + q] + np.dot(theta, eps[t + q - np.arange(1, q + 1)]) if q else eps[t + q]
        x[t] = c + ar_part + ma_part
    return TimeSeries(x[-n:], name=f"ARMA({p},{q})")


def random_walk(n: int, *, x0: float = 0.0, drift: float = 0.0, sigma: float = 1.0, rng=None) -> TimeSeries:
    """Simulate a random walk with optional drift, a canonical DS process."""
    r = _rng(rng)
    eps = r.normal(0.0, sigma, n)
    x = np.empty(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = x[t - 1] + drift + eps[t]
    return TimeSeries(x, name="RandomWalk")


def sarma(p: int, q: int, P: int, Q: int, s: int, phi, theta, seasonal_phi, seasonal_theta, n: int, *, c: float = 0.0, sigma: float = 1.0, rng=None) -> TimeSeries:
    """Simulate an additive seasonal ARMA process."""
    if s < 2:
        raise ValueError("seasonal period s must be at least 2")
    phi = np.asarray(list(phi), float); theta = np.asarray(list(theta), float)
    seasonal_phi = np.asarray(list(seasonal_phi), float); seasonal_theta = np.asarray(list(seasonal_theta), float)
    if phi.size != p or theta.size != q or seasonal_phi.size != P or seasonal_theta.size != Q:
        raise ValueError("coefficient lengths must match model orders")
    r = _rng(rng)
    maxlag = max(p, q, P * s, Q * s, 2) + 20
    eps = r.normal(0, sigma, n + maxlag + Q * s)
    x = np.zeros(n + maxlag)
    for t in range(maxlag, n + maxlag):
        ar = np.dot(phi, x[t - np.arange(1, p + 1)]) if p else 0.0
        sar = np.dot(seasonal_phi, x[t - s * np.arange(1, P + 1)]) if P else 0.0
        ma = eps[t + Q * s] + (np.dot(theta, eps[t + Q * s - np.arange(1, q + 1)]) if q else 0.0)
        sma = np.dot(seasonal_theta, eps[t + Q * s - s * np.arange(1, Q + 1)]) if Q else 0.0
        x[t] = c + ar + sar + ma + sma
    return TimeSeries(x[-n:], name=f"SARMA({p},{q})x({P},{Q})_{s}")


def simulate_process(kind: str, n: int, **kwargs) -> TimeSeries:
    """Unified EViews-like simulation dispatcher."""
    key = kind.lower().replace("-", "")
    if key in {"wn", "white_noise", "whitenoise"}:
        return white_noise(n, **kwargs)
    if key == "ar":
        return ar(n=n, **kwargs)
    if key == "ma":
        return ma(n=n, **kwargs)
    if key == "arma":
        return arma(n=n, **kwargs)
    if key in {"rw", "randomwalk", "random_walk"}:
        return random_walk(n, **kwargs)
    if key == "sarma":
        return sarma(n=n, **kwargs)
    raise ValueError(f"unsupported process kind: {kind}")
