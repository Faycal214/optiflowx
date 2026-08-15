"""Public API for discrete random variables from MSPRO Chapter 4."""

from __future__ import annotations

from collections.abc import Callable

from .conditional import RandomVariable as _RandomVariable


class RandomVariable(_RandomVariable):
    """Finite discrete random variable defined on a finite probability space.

    ``values[omega]`` gives the value of the variable at outcome ``omega``.
    """

    @property
    def support(self):
        """Return the distinct values taken by the random variable."""
        return tuple(dict.fromkeys(self.values.values()))

    def expected_value(self) -> float:
        """Return ``E[X]`` as an alias for ``expectation()``."""
        return self.expectation()

    def transform(self, function: Callable[[float], float], *, name: str | None = None):
        """Return the pointwise transformation ``g(X)``."""
        return RandomVariable(
            self.space,
            {outcome: float(function(value)) for outcome, value in self.values.items()},
            name=name,
        )

    def apply(self, function: Callable[[float], float], *, name: str | None = None):
        """Apply ``function`` pointwise and return a canonical ``RandomVariable``."""
        return self.transform(function, name=name)

    def _binary(self, other, operation):
        """Apply a binary operation while preserving the public RV type."""
        if isinstance(other, _RandomVariable):
            if other.space is not self.space:
                raise ValueError("random variables must belong to the same probability space")
            values = {
                outcome: float(operation(self.values[outcome], other.values[outcome]))
                for outcome in self.space.outcomes
            }
        else:
            values = {
                outcome: float(operation(self.values[outcome], float(other)))
                for outcome in self.space.outcomes
            }
        return RandomVariable(self.space, values)

    def __add__(self, other):
        """Return the pointwise sum ``X+Y`` or ``X+c``."""
        return self._binary(other, lambda a, b: a + b)

    def __sub__(self, other):
        """Return the pointwise difference ``X-Y`` or ``X-c``."""
        return self._binary(other, lambda a, b: a - b)

    def __mul__(self, other):
        """Return the pointwise product ``XY`` or ``cX``."""
        return self._binary(other, lambda a, b: a * b)

    def __radd__(self, other):
        """Return ``other + X``."""
        return self.__add__(other)

    def __rsub__(self, other):
        """Return ``other - X``."""
        return self._binary(other, lambda a, b: b - a)

    def __rmul__(self, other):
        """Return ``other * X``."""
        return self.__mul__(other)

    def __neg__(self):
        """Return the pointwise negation ``-X``."""
        return self.transform(lambda value: -value)


__all__ = ["RandomVariable"]
