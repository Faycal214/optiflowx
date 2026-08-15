"""Finite random variables from MSPRO Chapter 4."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Hashable

import numpy as np

Outcome = Hashable


@dataclass(frozen=True)
class RandomVariable:
    """A finite-valued random variable defined on a finite probability space.
    
    Mathematical object
    ------------------
    Public stochastic object exposed by the OptiFlowX API.
    
    Course basis
    ------------
    The implementation follows the corresponding MSPRO course material documented by OptiFlowX.
    
    Parameters
    ----------
    space : 'FiniteProbabilitySpace'
        Input argument.
    values : Mapping[Outcome, float]
        Values or mapping.
    name : str | None
        Optional display name.
    
    Examples
    --------
    See the executable examples for `random_variable.py` and the API reference."""

    space: "FiniteProbabilitySpace"
    values: Mapping[Outcome, float]
    name: str | None = None

    def __post_init__(self) -> None:
        if set(self.values) != set(self.space.outcomes):
            raise ValueError("values must be defined exactly on the probability-space outcomes")
        if any(not np.isfinite(float(v)) for v in self.values.values()):
            raise ValueError("random-variable values must be finite")

    @property
    def support(self) -> tuple[float, ...]:
        """Return the distinct values taken by the random variable.
        
        
        Returns
        -------
        tuple[float, ...]
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `support`."""
        return tuple(dict.fromkeys(float(v) for v in self.values.values()))

    def array(self) -> np.ndarray:
        """Return values in the probability-space outcome order.
        
        
        Returns
        -------
        np.ndarray
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `array`."""
        return np.asarray([self.values[o] for o in self.space.outcomes], dtype=float)

    def expectation(self) -> float:
        """Return ``E(X) = sum X(omega) P({omega})``.
        
        
        Returns
        -------
        float
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `expectation`."""
        return float(np.dot(self.array(), self.space.probabilities_array))

    def expected_value(self) -> float:
        """Return ``E(X)`` using the canonical API spelling.
        
        
        Returns
        -------
        float
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `expected_value`."""
        return self.expectation()

    def transform(self, function: Callable[[float], float], *, name: str | None = None) -> "RandomVariable":
        """Return the pointwise transformation ``g(X)``.
        
        Parameters
        ----------
        function : Callable[[float], float]
            Callable transformation.
        name : str | None
            Optional display name.
        
        Returns
        -------
        'RandomVariable'
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `transform`."""
        return RandomVariable(self.space, {o: float(function(v)) for o, v in self.values.items()}, name=name)

    def apply(self, function: Callable[[float], float], *, name: str | None = None) -> "RandomVariable":
        """Apply a scalar function pointwise to the random variable.
        
        Parameters
        ----------
        function : Callable[[float], float]
            Callable transformation.
        name : str | None
            Optional display name.
        
        Returns
        -------
        'RandomVariable'
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `apply`."""
        return self.transform(function, name=name)

    def _binary(self, other: float | "RandomVariable", op: Callable[[float, float], float]) -> "RandomVariable":
        if isinstance(other, RandomVariable):
            if other.space is not self.space:
                raise ValueError("random variables must belong to the same probability space")
            values = {o: float(op(self.values[o], other.values[o])) for o in self.space.outcomes}
        else:
            values = {o: float(op(self.values[o], float(other))) for o in self.space.outcomes}
        return RandomVariable(self.space, values)

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b)

    def __mul__(self, other):
        return self._binary(other, lambda a, b: a * b)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self._binary(other, lambda a, b: b - a)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self.transform(lambda x: -x)


__all__ = ["RandomVariable"]
