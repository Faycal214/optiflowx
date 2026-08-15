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


__all__ = ["RandomVariable"]
