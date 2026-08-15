"""Public API for finite probability spaces from MSPRO Chapter 4."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from .conditional import FiniteProbabilitySpace as _FiniteProbabilitySpace


class FiniteProbabilitySpace(_FiniteProbabilitySpace):
    """Finite probability space ``(Omega, P)`` for the discrete course framework.

    Parameters
    ----------
    outcomes:
        Unique elements of the finite sample space ``Omega``.
    probabilities:
        Point probabilities in the same order as ``outcomes``.
    """

    @property
    def n_outcomes(self) -> int:
        """Return the cardinality ``|Omega|`` of the sample space."""
        return len(self.outcomes)

    def probability_of(self, event: Iterable) -> float:
        """Return ``P(A)`` for a finite event ``A``."""
        return self.probability(event)

    def random_variable(self, values, *, name: str | None = None):
        """Create a canonical :class:`RandomVariable` on this space."""
        from .random_variable import RandomVariable

        if isinstance(values, Mapping):
            mapping = values
        else:
            if len(values) != len(self.outcomes):
                raise ValueError("values must have one entry per outcome")
            mapping = dict(zip(self.outcomes, values))
        return RandomVariable(self, mapping, name=name)

    def partition(self, blocks):
        """Create a canonical :class:`Partition` from its blocks."""
        from .partition import Partition

        return Partition.from_blocks(blocks, self)


__all__ = ["FiniteProbabilitySpace"]
