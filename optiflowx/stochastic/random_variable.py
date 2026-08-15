"""Public API for discrete random variables from MSPRO Chapter 4."""

from .conditional import RandomVariable as _RandomVariable


class RandomVariable(_RandomVariable):
    """Finite discrete random variable defined on a finite probability space."""

    @property
    def support(self):
        """Return the distinct values taken by the random variable."""
        return tuple(dict.fromkeys(self.values.values()))

    def expected_value(self) -> float:
        """Alias for ``expectation()`` using a conventional statistical name."""
        return self.expectation()


__all__ = ["RandomVariable"]
