"""Birth-death processes from the MSPRO Chapter 3 course."""

from __future__ import annotations

from collections.abc import Sequence
from math import comb, exp, factorial

import numpy as np

from .continuous_time_markov_chain import ContinuousTimeMarkovChain


class BirthDeathProcess:
    """Birth-death process with birth rates ``lambda_k`` and death rates ``mu_k``.
    
    Mathematical object
    ------------------
    Public stochastic object exposed by the OptiFlowX API.
    
    Course basis
    ------------
    The implementation follows the corresponding MSPRO course material documented by OptiFlowX.
    
    Parameters
    ----------
    birth_rates
        Birth-rate specification.
    death_rates
        Death-rate specification.
    max_state : int | None
        Optional upper state bound.
    
    Examples
    --------
    See the executable examples for `birth_death_process.py` and the API reference."""

    def __init__(self, birth_rates, death_rates, *, max_state: int | None = None):
        """Create a process from rate sequences or callable rate functions.
        
        Parameters
        ----------
        birth_rates
            Birth-rate specification.
        death_rates
            Death-rate specification.
        max_state : int | None
            Optional upper state bound.
        
        Raises
        ------
        ValueError
            Raised when an input or mathematical precondition is violated.
        
        Examples
        --------
        See the executable examples and API reference for `__init__`."""
        if max_state is not None and (isinstance(max_state, bool) or max_state < 0):
            raise ValueError("max_state must be a non-negative integer")
        self.max_state = max_state
        self._birth = birth_rates
        self._death = death_rates
        if max_state is not None:
            for k in range(max_state + 1):
                self._validate_rate(self.birth_rate(k))
                self._validate_rate(self.death_rate(k))

    @property
    def generator(self) -> np.ndarray:
        """Return the finite generator ``Q``.
        
        
        Returns
        -------
        np.ndarray
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `generator`."""
        return self.generator_matrix()

    @classmethod
    def finite(cls, birth_rates: Sequence[float], death_rates: Sequence[float]) -> "BirthDeathProcess":
        """Construct a finite-state process from two equally sized rate sequences.
        
        Parameters
        ----------
        birth_rates : Sequence[float]
            Birth-rate specification.
        death_rates : Sequence[float]
            Death-rate specification.
        
        Returns
        -------
        'BirthDeathProcess'
            Result produced by the operation.
        
        Raises
        ------
        ValueError
            Raised when an input or mathematical precondition is violated.
        
        Examples
        --------
        See the executable examples and API reference for `finite`."""
        if len(birth_rates) != len(death_rates) or not birth_rates:
            raise ValueError("birth_rates and death_rates must be non-empty and equally sized")
        return cls(birth_rates, death_rates, max_state=len(birth_rates) - 1)

    @classmethod
    def linear(cls, birth_rate: float, death_rate: float, *, immigration: float = 0.0, emigration: float = 0.0, max_state: int | None = None) -> "BirthDeathProcess":
        """Construct linear rates ``lambda_n=n*lambda+alpha`` and ``mu_n=n*mu+beta``.
        
        Parameters
        ----------
        birth_rate : float
            Linear birth-rate coefficient.
        death_rate : float
            Linear death-rate coefficient.
        immigration : float
            Immigration-rate offset.
        emigration : float
            Emigration-rate offset.
        max_state : int | None
            Optional upper state bound.
        
        Returns
        -------
        'BirthDeathProcess'
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `linear`."""
        for x in (birth_rate, death_rate, immigration, emigration):
            cls._validate_rate(x)
        return cls(
            lambda k: k * birth_rate + immigration,
            lambda k: 0.0 if k == 0 else k * death_rate + emigration,
            max_state=max_state,
        )

    @classmethod
    def pure_immigration(cls, rate: float) -> "BirthDeathProcess":
        """Construct the pure-immigration case.
        
        Parameters
        ----------
        rate : float
            Process rate.
        
        Returns
        -------
        'BirthDeathProcess'
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `pure_immigration`."""
        cls._validate_rate(rate)
        return cls(lambda _k: rate, lambda _k: 0.0)

    @classmethod
    def pure_birth(cls, rate: float) -> "BirthDeathProcess":
        """Construct the pure-birth case ``lambda_k=k*rate``.
        
        Parameters
        ----------
        rate : float
            Process rate.
        
        Returns
        -------
        'BirthDeathProcess'
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `pure_birth`."""
        cls._validate_rate(rate)
        return cls(lambda k: k * rate, lambda _k: 0.0)

    @classmethod
    def pure_death(cls, rate: float) -> "BirthDeathProcess":
        """Construct the pure-death case ``mu_k=k*rate``.
        
        Parameters
        ----------
        rate : float
            Process rate.
        
        Returns
        -------
        'BirthDeathProcess'
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `pure_death`."""
        cls._validate_rate(rate)
        return cls(lambda _k: 0.0, lambda k: k * rate)

    def birth_rate(self, k: int) -> float:
        """Return the birth rate ``lambda_k``.
        
        Parameters
        ----------
        k : int
            Count or order.
        
        Returns
        -------
        float
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `birth_rate`."""
        self._validate_state(k)
        return self._rate_value(self._birth, k)

    def death_rate(self, k: int) -> float:
        """Return the death rate ``mu_k``.
        
        Parameters
        ----------
        k : int
            Count or order.
        
        Returns
        -------
        float
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `death_rate`."""
        self._validate_state(k)
        return self._rate_value(self._death, k)

    def generator_matrix(self) -> np.ndarray:
        """Build the finite generator matrix ``Q``.
        
        
        Returns
        -------
        np.ndarray
            Result produced by the operation.
        
        Raises
        ------
        ValueError
            Raised when an input or mathematical precondition is violated.
        
        Examples
        --------
        See the executable examples and API reference for `generator_matrix`."""
        if self.max_state is None:
            raise ValueError("max_state is required to build a finite generator")
        n = self.max_state + 1
        q = np.zeros((n, n), dtype=float)
        for k in range(n):
            birth = 0.0 if k == self.max_state else self.birth_rate(k)
            death = 0.0 if k == 0 else self.death_rate(k)
            q[k, k] = -(birth + death)
            if k < self.max_state:
                q[k, k + 1] = birth
            if k > 0:
                q[k, k - 1] = death
        return q

    def to_ctmc(self) -> ContinuousTimeMarkovChain:
        """Return the corresponding finite-state homogeneous CTMC.
        
        
        Returns
        -------
        ContinuousTimeMarkovChain
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `to_ctmc`."""
        return ContinuousTimeMarkovChain(self.generator_matrix())

    def jump_chain_matrix(self) -> np.ndarray:
        """Return the embedded jump-chain matrix.
        
        
        Returns
        -------
        np.ndarray
            Result produced by the operation.
        
        Raises
        ------
        ValueError
            Raised when an input or mathematical precondition is violated.
        
        Examples
        --------
        See the executable examples and API reference for `jump_chain_matrix`."""
        if self.max_state is None:
            raise ValueError("max_state is required to build a finite jump chain")
        n = self.max_state + 1
        out = np.zeros((n, n))
        for k in range(n):
            birth = 0.0 if k == self.max_state else self.birth_rate(k)
            death = 0.0 if k == 0 else self.death_rate(k)
            total = birth + death
            if total == 0.0:
                out[k, k] = 1.0
            else:
                if k < self.max_state:
                    out[k, k + 1] = birth / total
                if k > 0:
                    out[k, k - 1] = death / total
        return out

    def jump_chain(self):
        """Return the embedded jump chain as a public ``MarkovChain``.
        
        
        Examples
        --------
        See the executable examples and API reference for `jump_chain`."""
        from .markov_chain import MarkovChain
        return MarkovChain(self.jump_chain_matrix())

    def kolmogorov_derivative(self, probabilities: Sequence[float]) -> np.ndarray:
        """Return the birth-death form of ``p'(t)=p(t)Q``.
        
        Parameters
        ----------
        probabilities : Sequence[float]
            Probability vector.
        
        Returns
        -------
        np.ndarray
            Result produced by the operation.
        
        Raises
        ------
        ValueError
            Raised when an input or mathematical precondition is violated.
        
        Examples
        --------
        See the executable examples and API reference for `kolmogorov_derivative`."""
        if self.max_state is None:
            raise ValueError("max_state is required for a finite probability vector")
        p = np.asarray(probabilities, dtype=float)
        if p.shape != (self.max_state + 1,) or np.any(p < 0) or not np.isclose(p.sum(), 1.0):
            raise ValueError("probabilities must sum to 1 and match max_state")
        d = np.zeros_like(p)
        for k in range(self.max_state + 1):
            birth_out = 0.0 if k == self.max_state else self.birth_rate(k)
            death_out = 0.0 if k == 0 else self.death_rate(k)
            birth_in = 0.0 if k == 0 else self.birth_rate(k - 1) * p[k - 1]
            death_in = 0.0 if k == self.max_state else self.death_rate(k + 1) * p[k + 1]
            d[k] = birth_in + death_in - (birth_out + death_out) * p[k]
        return d

    def stationary_weights(self, n_terms: int) -> np.ndarray:
        """Return the finite products used by the stationary law.
        
        Parameters
        ----------
        n_terms : int
            Number of truncated terms or states.
        
        Returns
        -------
        np.ndarray
            Result produced by the operation.
        
        Raises
        ------
        ValueError
            Raised when an input or mathematical precondition is violated.
        
        Examples
        --------
        See the executable examples and API reference for `stationary_weights`."""
        if isinstance(n_terms, bool) or n_terms < 0:
            raise ValueError("n_terms must be a non-negative integer")
        weights = np.ones(n_terms + 1)
        for k in range(1, n_terms + 1):
            mu = self.death_rate(k)
            if mu <= 0:
                raise ValueError("the course product formula requires positive mu_k")
            weights[k] = weights[k - 1] * self.birth_rate(k - 1) / mu
        return weights

    def stationary_weights_at(self, n_terms: int) -> np.ndarray:
        """Return stationary product weights through state ``n_terms``.
        
        Parameters
        ----------
        n_terms : int
            Number of truncated terms or states.
        
        Returns
        -------
        np.ndarray
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `stationary_weights_at`."""
        return self.stationary_weights(n_terms)

    def stationary_distribution(self, n_terms: int | None = None) -> np.ndarray:
        """Return the normalized stationary law in the finite/product setting.
        
        Parameters
        ----------
        n_terms : int | None, default None
            Number of truncated terms or states.
        
        Returns
        -------
        np.ndarray
            Result produced by the operation.
        
        Raises
        ------
        ValueError
            Raised when an input or mathematical precondition is violated.
        
        Examples
        --------
        See the executable examples and API reference for `stationary_distribution`."""
        if self.max_state is not None:
            return self.to_ctmc().stationary_distribution()
        if n_terms is None:
            raise ValueError("n_terms is required for the countable-state formula")
        weights = self.stationary_weights(n_terms)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("stationary weights cannot be normalized")
        return weights / total

    def pure_birth_reciprocal_rate_sum(self, n_terms: int) -> float:
        """Return the partial sum of ``1/lambda_k`` used in the explosion criterion.
        
        Parameters
        ----------
        n_terms : int
            Number of truncated terms or states.
        
        Returns
        -------
        float
            Result produced by the operation.
        
        Raises
        ------
        ValueError
            Raised when an input or mathematical precondition is violated.
        
        Examples
        --------
        See the executable examples and API reference for `pure_birth_reciprocal_rate_sum`."""
        if isinstance(n_terms, bool) or n_terms < 0:
            raise ValueError("n_terms must be a non-negative integer")
        rates = [self.birth_rate(k) for k in range(n_terms + 1)]
        if any(rate <= 0 for rate in rates):
            return float("inf")
        return float(sum(1.0 / rate for rate in rates))

    @staticmethod
    def pure_birth_probability(n: int, t: float, *, rate: float) -> float:
        """Return the pure-birth probability formula from the chapter.
        
        Parameters
        ----------
        n : int
            Integer index required by the operation.
        t : float
            Non-negative time.
        rate : float
            Process rate.
        
        Returns
        -------
        float
            Result produced by the operation.
        
        Raises
        ------
        ValueError
            Raised when an input or mathematical precondition is violated.
        
        Examples
        --------
        See the executable examples and API reference for `pure_birth_probability`."""
        if n < 1 or rate <= 0 or not np.isfinite(t) or t < 0:
            raise ValueError("invalid parameters")
        p = exp(-rate * float(t))
        return float(p * (1 - p) ** (n - 1))

    @staticmethod
    def pure_death_probability(n: int, t: float, *, initial_population: int, rate: float) -> float:
        """Return the pure-death binomial probability.
        
        Parameters
        ----------
        n : int
            Integer index required by the operation.
        t : float
            Non-negative time.
        initial_population : int
            Initial population size.
        rate : float
            Process rate.
        
        Returns
        -------
        float
            Result produced by the operation.
        
        Raises
        ------
        ValueError
            Raised when an input or mathematical precondition is violated.
        
        Examples
        --------
        See the executable examples and API reference for `pure_death_probability`."""
        if initial_population < 0 or n < 0 or n > initial_population or rate <= 0 or not np.isfinite(t) or t < 0:
            raise ValueError("invalid parameters")
        p = exp(-rate * float(t))
        return float(comb(initial_population, n) * p**n * (1 - p) ** (initial_population - n))

    @staticmethod
    def pure_immigration_probability(n: int, t: float, *, rate: float) -> float:
        """Return the Poisson law in the pure-immigration case.
        
        Parameters
        ----------
        n : int
            Integer index required by the operation.
        t : float
            Non-negative time.
        rate : float
            Process rate.
        
        Returns
        -------
        float
            Result produced by the operation.
        
        Raises
        ------
        ValueError
            Raised when an input or mathematical precondition is violated.
        
        Examples
        --------
        See the executable examples and API reference for `pure_immigration_probability`."""
        if n < 0 or rate < 0 or not np.isfinite(t) or t < 0:
            raise ValueError("invalid parameters")
        x = rate * float(t)
        return float(exp(-x) * x**n / factorial(n))

    def _validate_state(self, k: int) -> None:
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 0:
            raise ValueError("state must be a non-negative integer")
        if self.max_state is not None and k > self.max_state:
            raise ValueError("state exceeds max_state")

    @staticmethod
    def _rate_value(rate, k: int) -> float:
        value = float(rate(k) if callable(rate) else rate[k])
        if not np.isfinite(value) or value < 0:
            raise ValueError("rates must be finite and non-negative")
        return value

    @staticmethod
    def _validate_rate(value: float) -> None:
        if not np.isfinite(value) or value < 0:
            raise ValueError("rates must be finite and non-negative")


__all__ = ["BirthDeathProcess"]
