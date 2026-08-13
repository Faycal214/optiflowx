"""Poisson processes covered by the MSPRO Processus Aléatoires course."""

from __future__ import annotations

from collections.abc import Callable
from math import exp, factorial
from typing import Sequence

import numpy as np


class PoissonProcess:
    """Homogeneous Poisson process with rate lambda > 0."""

    def __init__(self, rate: float) -> None:
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("rate must be strictly positive and finite")
        self.rate = float(rate)

    def count_probability(self, n: int, t: float) -> float:
        """Return P(N(t)=n) = exp(-lambda*t)(lambda*t)^n/n!."""
        self._validate_time(t)
        self._validate_count(n)
        x = self.rate * float(t)
        return float(exp(-x) * x**int(n) / factorial(int(n)))

    def increment_probability(self, n: int, s: float, t: float) -> float:
        """Return P(N(t)-N(s)=n), for 0 <= s <= t."""
        self._validate_interval(s, t)
        return self._count_probability_for_mean(n, self.rate * (float(t) - float(s)))

    def interarrival_samples(
        self,
        n: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Generate n IID exponential inter-arrival times with rate lambda."""
        self._validate_positive_count(n)
        generator = rng if rng is not None else np.random.default_rng()
        return generator.exponential(scale=1.0 / self.rate, size=int(n))

    def arrival_times(
        self,
        n: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Generate the first n occurrence times tau_1,...,tau_n."""
        return np.cumsum(self.interarrival_samples(n, rng=rng))

    def simulate(
        self,
        t_max: float,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Generate all occurrence times in [0, t_max]."""
        self._validate_time(t_max)
        generator = rng if rng is not None else np.random.default_rng()
        times: list[float] = []
        current = 0.0
        while True:
            current += float(generator.exponential(scale=1.0 / self.rate))
            if current > float(t_max):
                break
            times.append(current)
        return np.asarray(times, dtype=float)

    def count_sample(
        self,
        t: float,
        *,
        rng: np.random.Generator | None = None,
    ) -> int:
        """Sample N(t) directly from its Poisson(lambda*t) law."""
        self._validate_time(t)
        generator = rng if rng is not None else np.random.default_rng()
        return int(generator.poisson(self.rate * float(t)))

    def conditional_first_arrival_cdf(self, y: float, s: float) -> float:
        """Return P(tau_1 <= y | N(s)=1), which is Uniform([0,s])."""
        self._validate_positive_time(s, "s")
        if y <= 0:
            return 0.0
        if y >= s:
            return 1.0
        return float(y / s)

    def conditional_arrival_times(
        self,
        k: int,
        s: float,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Sample occurrence times conditional on N(s)=k.

        The k occurrence times are an ordered k-sample from Uniform([0,s]).
        """
        self._validate_positive_count(k)
        self._validate_positive_time(s, "s")
        generator = rng if rng is not None else np.random.default_rng()
        return np.sort(generator.uniform(0.0, float(s), size=int(k)))

    def superpose(self, other: "PoissonProcess") -> "PoissonProcess":
        """Return the superposition of two independent Poisson processes."""
        if not isinstance(other, PoissonProcess):
            raise TypeError("other must be a PoissonProcess")
        return PoissonProcess(self.rate + other.rate)

    def split(self, probability: float) -> tuple["PoissonProcess", "PoissonProcess"]:
        """Split the process using independent Bernoulli(p) labels."""
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("probability must lie in [0, 1]")
        return (
            PoissonProcess(self.rate * float(probability)) if probability > 0 else _DegeneratePoissonProcess(),
            PoissonProcess(self.rate * (1.0 - float(probability))) if probability < 1 else _DegeneratePoissonProcess(),
        )

    @staticmethod
    def _count_probability_for_mean(n: int, mean: float) -> float:
        PoissonProcess._validate_count(n)
        return float(exp(-mean) * mean**int(n) / factorial(int(n)))

    @staticmethod
    def _validate_count(n: int) -> None:
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError("n must be a non-negative integer")

    @staticmethod
    def _validate_positive_count(n: int) -> None:
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("n must be a positive integer")

    @staticmethod
    def _validate_time(t: float) -> None:
        if not np.isfinite(t) or t < 0:
            raise ValueError("time must be finite and non-negative")

    @staticmethod
    def _validate_positive_time(t: float, name: str) -> None:
        if not np.isfinite(t) or t <= 0:
            raise ValueError(f"{name} must be strictly positive and finite")

    @staticmethod
    def _validate_interval(s: float, t: float) -> None:
        PoissonProcess._validate_time(s)
        PoissonProcess._validate_time(t)
        if s > t:
            raise ValueError("require s <= t")


class NonHomogeneousPoissonProcess:
    """Non-homogeneous Poisson process defined by an intensity lambda(t)."""

    def __init__(
        self,
        intensity: Callable[[float], float],
        mean_function: Callable[[float], float] | None = None,
    ) -> None:
        if not callable(intensity):
            raise TypeError("intensity must be callable")
        self.intensity = intensity
        self.mean_function = mean_function

    def mean(self, t: float) -> float:
        """Return m(t)=integral_0^t lambda(x) dx.

        A cumulative mean function may be supplied explicitly. Otherwise a
        numerical trapezoidal integral is used for evaluation.
        """
        if not np.isfinite(t) or t < 0:
            raise ValueError("time must be finite and non-negative")
        t = float(t)
        if self.mean_function is not None:
            value = float(self.mean_function(t))
        else:
            if t == 0:
                return 0.0
            grid = np.linspace(0.0, t, 2049)
            values = np.asarray([self.intensity(x) for x in grid], dtype=float)
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError("intensity must return finite non-negative values")
            value = float(np.trapezoid(values, grid))
        if not np.isfinite(value) or value < 0:
            raise ValueError("mean function must be finite and non-negative")
        return value

    def count_probability(self, n: int, t: float) -> float:
        """Return the Poisson count probability with parameter m(t)."""
        PoissonProcess._validate_count(n)
        mean = self.mean(t)
        return PoissonProcess._count_probability_for_mean(n, mean)

    def increment_probability(self, n: int, s: float, t: float) -> float:
        """Return the increment count probability with mean m(t)-m(s)."""
        PoissonProcess._validate_interval(s, t)
        mean = self.mean(t) - self.mean(s)
        if mean < -1e-10:
            raise ValueError("mean function must be non-decreasing")
        return PoissonProcess._count_probability_for_mean(n, max(0.0, mean))


class _DegeneratePoissonProcess:
    """Internal zero-rate process produced by splitting at p=0 or p=1."""

    rate = 0.0

    def count_probability(self, n: int, t: float) -> float:
        PoissonProcess._validate_time(t)
        PoissonProcess._validate_count(n)
        return 1.0 if n == 0 else 0.0
