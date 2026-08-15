"""Public API for discrete-time stopping times from MSPRO Chapter 5."""

from __future__ import annotations

from typing import Mapping

from .martingale import StoppingTime as _StoppingTime


class StoppingTime(_StoppingTime):
    """Stopping time taking values in ``N union {infinity}``.

    The defining condition is ``{T=n} in F_n`` for every discrete time level.
    """

    @property
    def maximum_time(self):
        """Return the largest finite stopping value, if one exists."""
        finite = [v for v in self.values.values() if not __import__('math').isinf(v)]
        return max(finite) if finite else None

    @classmethod
    def from_values(cls, space, values: Mapping, filtration):
        """Create a stopping time from outcome-wise stopping values."""
        return cls(space, dict(values), filtration)

    def minimum(self, other):
        """Return ``T wedge S``."""
        self._same(other)
        return StoppingTime(
            self.space,
            {o: min(self.values[o], other.values[o]) for o in self.space.outcomes},
            self.filtration,
        )

    def maximum(self, other):
        """Return ``T vee S``."""
        self._same(other)
        return StoppingTime(
            self.space,
            {o: max(self.values[o], other.values[o]) for o in self.space.outcomes},
            self.filtration,
        )

    def add(self, other):
        """Return the pointwise sum, preserving infinity."""
        self._same(other)
        import numpy as np

        values = {
            o: np.inf
            if np.isinf(self.values[o]) or np.isinf(other.values[o])
            else self.values[o] + other.values[o]
            for o in self.space.outcomes
        }
        return StoppingTime(self.space, values, self.filtration)


__all__ = ["StoppingTime"]
