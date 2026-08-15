"""Public API for finite probability spaces from MSPRO Chapter 4."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

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

    def conditional_expectation(self, x, partition):
        """Return the canonical random variable ``E(X|G)``."""
        if x.space is not self:
            raise ValueError("random variable belongs to another probability space")
        values = {}
        for block in partition.blocks:
            mass = self.probability(block)
            if mass <= 0:
                raise ValueError("conditional expectation is undefined on a zero-probability block")
            mean = sum(self.probabilities[o] * x.values[o] for o in block) / mass
            for outcome in block:
                values[outcome] = float(mean)
        return self.random_variable(values, name=f"E({x.name or 'X'}|G)")

    def conditional_expectation_given(self, x, y):
        """Return the canonical ``E(X|Y)``."""
        from .partition import Partition

        return self.conditional_expectation(x, Partition.generated_by(y))

    def conditional_probability(self, event: Iterable, partition):
        """Return the canonical random variable ``P(A|G)``."""
        event = set(event)
        indicator = self.random_variable(
            {outcome: float(outcome in event) for outcome in self.outcomes},
            name="1_A",
        )
        result = self.conditional_expectation(indicator, partition)
        return self.random_variable(result.values, name="P(A|G)")

    def conditional_variance(self, x, partition):
        """Return the canonical ``Var(X|G)``."""
        ex = self.conditional_expectation(x, partition)
        return self.conditional_expectation(x * x, partition) - ex * ex

    def conditional_covariance(self, x, y, partition):
        """Return the canonical ``Cov(X,Y|G)``."""
        return self.conditional_expectation(x * y, partition) - (
            self.conditional_expectation(x, partition)
            * self.conditional_expectation(y, partition)
        )


__all__ = ["FiniteProbabilitySpace"]
