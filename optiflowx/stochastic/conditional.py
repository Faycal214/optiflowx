"""Finite conditional-expectation objects from MSPRO Chapter 4.

The chapter develops conditional expectation in a discrete setting. The
package therefore represents a finite probability space explicitly, models a
finite sigma-field by a partition, and represents a discrete random variable
by its value on each outcome.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Hashable

import numpy as np

Outcome = Hashable


@dataclass(frozen=True)
class RandomVariable:
    """A finite-valued random variable defined on a ``FiniteProbabilitySpace``."""

    space: "FiniteProbabilitySpace"
    values: Mapping[Outcome, float]
    name: str | None = None

    def __post_init__(self) -> None:
        if set(self.values) != set(self.space.outcomes):
            raise ValueError("values must be defined exactly on the probability-space outcomes")
        if any(not np.isfinite(float(v)) for v in self.values.values()):
            raise ValueError("random-variable values must be finite")

    def array(self) -> np.ndarray:
        """Return values in the probability-space outcome order."""
        return np.asarray([self.values[o] for o in self.space.outcomes], dtype=float)

    def expectation(self) -> float:
        """Return ``E(X) = sum X(omega) P({omega})``."""
        return float(np.dot(self.array(), self.space.probabilities_array))

    def apply(self, function: Callable[[float], float], *, name: str | None = None) -> "RandomVariable":
        """Apply a scalar function pointwise to the random variable."""
        return RandomVariable(self.space, {o: float(function(v)) for o, v in self.values.items()}, name=name)

    def _binary(self, other: float | "RandomVariable", op: Callable[[float, float], float]) -> "RandomVariable":
        if isinstance(other, RandomVariable):
            if other.space is not self.space:
                raise ValueError("random variables must belong to the same probability space")
            values = {o: op(self.values[o], other.values[o]) for o in self.space.outcomes}
        else:
            values = {o: op(self.values[o], float(other)) for o in self.space.outcomes}
        return RandomVariable(self.space, values)

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b)

    def __mul__(self, other):
        return self._binary(other, lambda a, b: a * b)

    def __neg__(self):
        return self.apply(lambda x: -x)


@dataclass(frozen=True)
class Partition:
    """A finite partition representing a finite sigma-field."""

    blocks: tuple[frozenset[Outcome], ...]

    @classmethod
    def from_blocks(
        cls, blocks: Iterable[Iterable[Outcome]], space: "FiniteProbabilitySpace"
    ) -> "Partition":
        """Build a partition after checking coverage and disjointness."""
        normalized = tuple(frozenset(block) for block in blocks)
        if not normalized or any(not block for block in normalized):
            raise ValueError("partition blocks must be non-empty")
        if set().union(*normalized) != set(space.outcomes):
            raise ValueError("partition must cover the probability space")
        if sum(map(len, normalized)) != len(space.outcomes):
            raise ValueError("partition blocks must be disjoint")
        return cls(normalized)

    @classmethod
    def generated_by(cls, rv: RandomVariable) -> "Partition":
        """Return the partition whose blocks are the level sets of ``rv``."""
        groups: dict[float, set[Outcome]] = {}
        for outcome, value in rv.values.items():
            groups.setdefault(float(value), set()).add(outcome)
        return cls.from_blocks(groups.values(), rv.space)

    def refines(self, coarser: "Partition") -> bool:
        """Return whether every block here lies inside a block of ``coarser``."""
        return all(any(block.issubset(parent) for parent in coarser.blocks) for block in self.blocks)


class FiniteProbabilitySpace:
    """Finite probability space used for the chapter's discrete framework.

    Outcomes and their probabilities are stored explicitly. This makes the
    weighted sums defining expectation and conditional expectation exact up to
    ordinary floating-point arithmetic.
    """

    def __init__(self, outcomes: Sequence[Outcome], probabilities: Sequence[float]) -> None:
        """Create ``(Omega, P)`` from its outcomes and point probabilities."""
        self.outcomes = tuple(outcomes)
        if not self.outcomes or len(set(self.outcomes)) != len(self.outcomes):
            raise ValueError("outcomes must be unique and non-empty")
        p = np.asarray(probabilities, dtype=float)
        if (
            p.shape != (len(self.outcomes),)
            or not np.all(np.isfinite(p))
            or np.any(p < 0)
            or not np.isclose(p.sum(), 1.0)
        ):
            raise ValueError("probabilities must be non-negative and sum to 1")
        self.probabilities = dict(zip(self.outcomes, p.astype(float)))

    @property
    def probabilities_array(self) -> np.ndarray:
        """Return point probabilities in outcome order."""
        return np.asarray([self.probabilities[o] for o in self.outcomes])

    def random_variable(
        self,
        values: Mapping[Outcome, float] | Sequence[float],
        *,
        name: str | None = None,
    ) -> RandomVariable:
        """Create a discrete random variable on this space."""
        if isinstance(values, Mapping):
            mapping = values
        else:
            if len(values) != len(self.outcomes):
                raise ValueError("values must have one entry per outcome")
            mapping = dict(zip(self.outcomes, values))
        return RandomVariable(self, mapping, name=name)

    def probability(self, event: Iterable[Outcome]) -> float:
        """Return the probability of a finite event."""
        event = set(event)
        if not event.issubset(self.outcomes):
            raise ValueError("event contains an unknown outcome")
        return float(sum(self.probabilities[o] for o in event))

    def conditional_probability_given_event(
        self, event: Iterable[Outcome], condition: Iterable[Outcome]
    ) -> float:
        """Return ``P(A|B)=P(A intersect B)/P(B)`` when ``P(B)>0``."""
        a = set(event)
        b = set(condition)
        if not a.issubset(self.outcomes) or not b.issubset(self.outcomes):
            raise ValueError("event contains an unknown outcome")
        denominator = self.probability(b)
        if denominator <= 0:
            raise ValueError("conditioning event must have positive probability")
        return self.probability(a & b) / denominator

    def conditional_expectation_given_event(
        self, x: RandomVariable, condition: Iterable[Outcome]
    ) -> float:
        """Return ``E(X|B)=E(X 1_B)/P(B)`` for ``P(B)>0``."""
        if x.space is not self:
            raise ValueError("random variable belongs to another probability space")
        b = set(condition)
        denominator = self.probability(b)
        if denominator <= 0:
            raise ValueError("conditioning event must have positive probability")
        numerator = sum(self.probabilities[o] * x.values[o] for o in b)
        return float(numerator / denominator)

    def partition(self, blocks: Iterable[Iterable[Outcome]]) -> Partition:
        """Create a finite sigma-field representation from its partition blocks."""
        return Partition.from_blocks(blocks, self)

    def conditional_expectation(self, x: RandomVariable, partition: Partition) -> RandomVariable:
        """Compute ``E(X | G)`` block-by-block.

        On a block ``G`` with positive probability, the chapter's discrete
        definition is the weighted mean
        ``sum_{omega in G} X(omega)P({omega}) / P(G)``. The result is constant
        on each block and is therefore measurable with respect to ``G``.
        """
        if x.space is not self:
            raise ValueError("random variable belongs to another probability space")
        values = {}
        for block in partition.blocks:
            mass = self.probability(block)
            if mass <= 0:
                raise ValueError("conditional expectation is undefined on a zero-probability block")
            mean = sum(self.probabilities[o] * x.values[o] for o in block) / mass
            for o in block:
                values[o] = float(mean)
        return RandomVariable(self, values, name=f"E({x.name or 'X'}|G)")

    def conditional_expectation_given(self, x: RandomVariable, y: RandomVariable) -> RandomVariable:
        """Compute ``E(X | Y)`` through the partition generated by ``Y``."""
        return self.conditional_expectation(x, Partition.generated_by(y))

    def conditional_probability(self, event: Iterable[Outcome], partition: Partition) -> RandomVariable:
        """Return ``P(A|G)`` as ``E(1_A|G)``."""
        event = set(event)
        indicator = self.random_variable(
            [float(o in event) for o in self.outcomes], name="1_A"
        )
        result = self.conditional_expectation(indicator, partition)
        return RandomVariable(self, result.values, name="P(A|G)")

    def total_expectation(self, x: RandomVariable, partition: Partition) -> float:
        """Return ``E[E(X|G)]`` (the total-expectation identity)."""
        return self.conditional_expectation(x, partition).expectation()

    def tower(self, x: RandomVariable, finer: Partition, coarser: Partition) -> RandomVariable:
        """Apply the tower property when the first partition refines the second."""
        if not finer.refines(coarser):
            raise ValueError("finer partition must refine coarser partition")
        return self.conditional_expectation(
            self.conditional_expectation(x, finer), coarser
        )

    def pull_out(self, y: RandomVariable, x: RandomVariable, partition: Partition) -> RandomVariable:
        """Return the residual of ``E(YX|G)=Y E(X|G)``."""
        if not self.check_measurable(y, partition):
            raise ValueError("y must be measurable with respect to the conditioning partition")
        left = self.conditional_expectation(y * x, partition)
        right = y * self.conditional_expectation(x, partition)
        return left - right

    def conditional_variance(self, x: RandomVariable, partition: Partition) -> RandomVariable:
        """Return ``Var(X|G)=E(X^2|G)-E(X|G)^2``."""
        ex = self.conditional_expectation(x, partition)
        return self.conditional_expectation(x * x, partition) - ex * ex

    def conditional_covariance(
        self, x: RandomVariable, y: RandomVariable, partition: Partition
    ) -> RandomVariable:
        """Return ``Cov(X,Y|G)`` from the chapter's formula."""
        return self.conditional_expectation(x * y, partition) - (
            self.conditional_expectation(x, partition)
            * self.conditional_expectation(y, partition)
        )

    def variance(self, x: RandomVariable) -> float:
        """Return the ordinary variance of ``x``."""
        m = x.expectation()
        return float(((x - m) * (x - m)).expectation())

    def covariance(self, x: RandomVariable, y: RandomVariable) -> float:
        """Return the ordinary covariance of ``x`` and ``y``."""
        return float((x * y).expectation() - x.expectation() * y.expectation())

    def total_variance(self, x: RandomVariable, partition: Partition) -> float:
        """Return the total-variance decomposition from the chapter."""
        ex = self.conditional_expectation(x, partition)
        return float(self.variance(ex) + self.conditional_variance(x, partition).expectation())

    def total_covariance(self, x: RandomVariable, y: RandomVariable, partition: Partition) -> float:
        """Return the total-covariance decomposition from the chapter."""
        ex = self.conditional_expectation(x, partition)
        ey = self.conditional_expectation(y, partition)
        return float(
            self.conditional_covariance(x, y, partition).expectation()
            + self.covariance(ex, ey)
        )

    def l2_projection(self, x: RandomVariable, partition: Partition) -> RandomVariable:
        """Return the chapter's ``L2`` projection interpretation of ``E(X|G)``."""
        if self.variance(x) < -1e-12:
            raise ValueError("invalid variance")
        return self.conditional_expectation(x, partition)

    def check_measurable(
        self, x: RandomVariable, partition: Partition, *, atol: float = 1e-12
    ) -> bool:
        """Check whether ``x`` is constant on every block of the partition."""
        return all(np.ptp([x.values[o] for o in block]) <= atol for block in partition.blocks)
