"""Finite probability-space tools for conditional expectation in Chapter 4."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Hashable

import numpy as np

Outcome = Hashable


@dataclass(frozen=True)
class RandomVariable:
    space: "FiniteProbabilitySpace"
    values: Mapping[Outcome, float]
    name: str | None = None

    def __post_init__(self) -> None:
        if set(self.values) != set(self.space.outcomes):
            raise ValueError("values must be defined exactly on the probability-space outcomes")
        if any(not np.isfinite(float(v)) for v in self.values.values()):
            raise ValueError("random-variable values must be finite")

    def array(self) -> np.ndarray:
        return np.asarray([self.values[o] for o in self.space.outcomes], dtype=float)

    def expectation(self) -> float:
        return float(np.dot(self.array(), self.space.probabilities_array))

    def apply(self, function: Callable[[float], float], *, name: str | None = None) -> "RandomVariable":
        return RandomVariable(self.space, {o: float(function(v)) for o, v in self.values.items()}, name=name)

    def _binary(self, other: float | "RandomVariable", op: Callable[[float, float], float]) -> "RandomVariable":
        if isinstance(other, RandomVariable):
            if other.space is not self.space:
                raise ValueError("random variables must belong to the same probability space")
            values = {o: op(self.values[o], other.values[o]) for o in self.space.outcomes}
        else:
            values = {o: op(self.values[o], float(other)) for o in self.space.outcomes}
        return RandomVariable(self.space, values)

    def __add__(self, other): return self._binary(other, lambda a, b: a + b)
    def __sub__(self, other): return self._binary(other, lambda a, b: a - b)
    def __mul__(self, other): return self._binary(other, lambda a, b: a * b)
    def __neg__(self): return self.apply(lambda x: -x)


@dataclass(frozen=True)
class Partition:
    blocks: tuple[frozenset[Outcome], ...]

    @classmethod
    def from_blocks(cls, blocks: Iterable[Iterable[Outcome]], space: "FiniteProbabilitySpace") -> "Partition":
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
        groups: dict[float, set[Outcome]] = {}
        for outcome, value in rv.values.items():
            groups.setdefault(float(value), set()).add(outcome)
        return cls.from_blocks(groups.values(), rv.space)

    def refines(self, coarser: "Partition") -> bool:
        return all(any(block.issubset(parent) for parent in coarser.blocks) for block in self.blocks)


class FiniteProbabilitySpace:
    """Finite probability space used for exact discrete conditional expectations."""

    def __init__(self, outcomes: Sequence[Outcome], probabilities: Sequence[float]) -> None:
        self.outcomes = tuple(outcomes)
        if not self.outcomes or len(set(self.outcomes)) != len(self.outcomes):
            raise ValueError("outcomes must be unique and non-empty")
        p = np.asarray(probabilities, dtype=float)
        if p.shape != (len(self.outcomes),) or not np.all(np.isfinite(p)) or np.any(p < 0) or not np.isclose(p.sum(), 1.0):
            raise ValueError("probabilities must be non-negative and sum to 1")
        self.probabilities = dict(zip(self.outcomes, p.astype(float)))

    @property
    def probabilities_array(self) -> np.ndarray:
        return np.asarray([self.probabilities[o] for o in self.outcomes])

    def random_variable(self, values: Mapping[Outcome, float] | Sequence[float], *, name: str | None = None) -> RandomVariable:
        if isinstance(values, Mapping):
            mapping = values
        else:
            if len(values) != len(self.outcomes):
                raise ValueError("values must have one entry per outcome")
            mapping = dict(zip(self.outcomes, values))
        return RandomVariable(self, mapping, name=name)

    def probability(self, event: Iterable[Outcome]) -> float:
        event = set(event)
        if not event.issubset(self.outcomes):
            raise ValueError("event contains an unknown outcome")
        return float(sum(self.probabilities[o] for o in event))

    def partition(self, blocks: Iterable[Iterable[Outcome]]) -> Partition:
        return Partition.from_blocks(blocks, self)

    def conditional_expectation(self, x: RandomVariable, partition: Partition) -> RandomVariable:
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
        return self.conditional_expectation(x, Partition.generated_by(y))

    def conditional_probability(self, event: Iterable[Outcome], partition: Partition) -> RandomVariable:
        event = set(event)
        indicator = self.random_variable([float(o in event) for o in self.outcomes], name="1_A")
        result = self.conditional_expectation(indicator, partition)
        return RandomVariable(self, result.values, name="P(A|G)")

    def total_expectation(self, x: RandomVariable, partition: Partition) -> float:
        return self.conditional_expectation(x, partition).expectation()

    def tower(self, x: RandomVariable, finer: Partition, coarser: Partition) -> RandomVariable:
        if not finer.refines(coarser):
            raise ValueError("finer partition must refine coarser partition")
        return self.conditional_expectation(self.conditional_expectation(x, finer), coarser)

    def pull_out(self, y: RandomVariable, x: RandomVariable, partition: Partition) -> RandomVariable:
        if not self.check_measurable(y, partition):
            raise ValueError("y must be measurable with respect to the conditioning partition")
        left = self.conditional_expectation(y * x, partition)
        right = y * self.conditional_expectation(x, partition)
        return left - right

    def conditional_variance(self, x: RandomVariable, partition: Partition) -> RandomVariable:
        ex = self.conditional_expectation(x, partition)
        return self.conditional_expectation(x * x, partition) - ex * ex

    def conditional_covariance(self, x: RandomVariable, y: RandomVariable, partition: Partition) -> RandomVariable:
        return self.conditional_expectation(x * y, partition) - self.conditional_expectation(x, partition) * self.conditional_expectation(y, partition)

    def variance(self, x: RandomVariable) -> float:
        m = x.expectation()
        return float(((x - m) * (x - m)).expectation())

    def covariance(self, x: RandomVariable, y: RandomVariable) -> float:
        return float((x * y).expectation() - x.expectation() * y.expectation())

    def total_variance(self, x: RandomVariable, partition: Partition) -> float:
        ex = self.conditional_expectation(x, partition)
        return float(self.variance(ex) + self.conditional_variance(x, partition).expectation())

    def total_covariance(self, x: RandomVariable, y: RandomVariable, partition: Partition) -> float:
        ex = self.conditional_expectation(x, partition)
        ey = self.conditional_expectation(y, partition)
        return float(self.conditional_covariance(x, y, partition).expectation() + self.covariance(ex, ey))

    def l2_projection(self, x: RandomVariable, partition: Partition) -> RandomVariable:
        """Projection of X onto the finite-dimensional L2 space measurable by G."""
        if self.variance(x) < -1e-12:
            raise ValueError("invalid variance")
        return self.conditional_expectation(x, partition)

    def check_measurable(self, x: RandomVariable, partition: Partition, *, atol: float = 1e-12) -> bool:
        return all(np.ptp([x.values[o] for o in block]) <= atol for block in partition.blocks)
