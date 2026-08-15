"""Uniformization (Jensen's method) for finite-state CTMCs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import poisson

from ..exceptions import NumericalError
from ..validation import DEFAULT_TOLERANCE, normalize_stochastic_matrix, validate_generator, validate_tolerance


def uniformization_transition_matrix(
    generator: Sequence[Sequence[float]],
    time: float,
    *,
    rate: float | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    max_terms: int = 100_000,
) -> np.ndarray:
    """Compute ``P(t)=exp(tQ)`` by uniformization.

    For ``nu >= max_i(-q_ii)``, define ``R = I + Q / nu``. Then

        P(t) = exp(-nu t) * sum_k (nu t)^k / k! * R^k.

    The Poisson series is truncated so that its omitted tail is at most
    ``tolerance``. ``max_terms`` prevents unexpectedly expensive expansions.
    """
    tol = validate_tolerance(tolerance)
    if not np.isfinite(time) or time < 0.0:
        raise ValueError("time must be finite and non-negative")
    if isinstance(max_terms, bool) or not isinstance(max_terms, (int, np.integer)) or max_terms < 1:
        raise ValueError("max_terms must be a positive integer")

    q = validate_generator(generator, tolerance=tol)
    n = q.shape[0]
    if time == 0.0:
        return np.eye(n)

    exit_rates = -np.diag(q)
    required_rate = float(np.max(exit_rates))
    if required_rate <= tol:
        return np.eye(n)

    if rate is None:
        nu = required_rate
    else:
        nu = float(rate)
        if not np.isfinite(nu) or nu <= 0.0:
            raise ValueError("uniformization rate must be finite and strictly positive")
        if nu + tol < required_rate:
            raise ValueError("uniformization rate must be at least the maximum exit rate")

    kernel = np.eye(n) + q / nu
    kernel = normalize_stochastic_matrix(kernel, tolerance=tol)
    mean = nu * float(time)

    cutoff = poisson.ppf(1.0 - tol, mean)
    if not np.isfinite(cutoff):
        raise NumericalError("uniformization truncation index is not finite")
    last = int(cutoff)
    if last + 1 > max_terms:
        raise NumericalError(
            "uniformization requires more terms than max_terms; "
            "increase max_terms or use another numerical method"
        )

    k_values = np.arange(last + 1, dtype=int)
    weights = poisson.pmf(k_values, mean)
    if not np.all(np.isfinite(weights)):
        raise NumericalError("uniformization Poisson weights are not finite")

    result = np.zeros((n, n), dtype=float)
    power = np.eye(n)
    for weight in weights:
        if weight:
            result += float(weight) * power
        power = power @ kernel

    return normalize_stochastic_matrix(result, tolerance=max(tol, 1e-14))


__all__ = ["uniformization_transition_matrix"]
