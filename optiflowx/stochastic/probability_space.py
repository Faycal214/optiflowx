"""Public API for finite probability spaces from MSPRO Chapter 4."""

from .conditional import FiniteProbabilitySpace as _FiniteProbabilitySpace


class FiniteProbabilitySpace(_FiniteProbabilitySpace):
    """Finite probability space ``(Omega, P)`` for the discrete course framework."""

    @property
    def n_outcomes(self) -> int:
        """Return the cardinality of the finite sample space."""
        return len(self.outcomes)

    def probability_of(self, event):
        """Return ``P(A)`` for the finite event ``A``."""
        return self.probability(event)


__all__ = ["FiniteProbabilitySpace"]
