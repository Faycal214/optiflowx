"""Public API for finite probability spaces from MSPRO Chapter 4."""

from collections.abc import Mapping, Sequence

from .conditional import FiniteProbabilitySpace as _FiniteProbabilitySpace
from .partition import Partition
from .random_variable import RandomVariable


class FiniteProbabilitySpace(_FiniteProbabilitySpace):
    """Finite probability space ``(Omega, P)`` for the discrete course framework."""

    @property
    def n_outcomes(self) -> int:
        """Return the cardinality of the finite sample space."""
        return len(self.outcomes)

    def probability_of(self, event):
        """Return ``P(A)`` for the finite event ``A``."""
        return self.probability(event)

    def random_variable(
        self,
        values: Mapping | Sequence[float],
        *,
        name: str | None = None,
    ) -> RandomVariable:
        """Create a public ``RandomVariable`` bound to this probability space."""
        if isinstance(values, Mapping):
            mapping = values
        else:
            if len(values) != self.n_outcomes:
                raise ValueError("values must have one entry per outcome")
            mapping = dict(zip(self.outcomes, values))
        return RandomVariable(self, mapping, name=name)

    def partition(self, blocks) -> Partition:
        """Create a public ``Partition`` representing a finite sigma-field."""
        return Partition.from_blocks(blocks, self)


__all__ = ["FiniteProbabilitySpace"]
