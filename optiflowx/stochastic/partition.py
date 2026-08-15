"""Finite partitions used as sigma-fields in MSPRO Chapter 4."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Hashable

Outcome = Hashable


@dataclass(frozen=True)
class Partition:
    """A finite partition representing a finite sigma-field."""

    blocks: tuple[frozenset[Outcome], ...]

    @property
    def n_blocks(self) -> int:
        """Return the number of blocks in the partition."""
        return len(self.blocks)

    def contains(self, outcome: Outcome) -> int:
        """Return the index of the block containing ``outcome``."""
        for index, block in enumerate(self.blocks):
            if outcome in block:
                return index
        raise ValueError(f"unknown outcome: {outcome!r}")

    @classmethod
    def from_blocks(cls, blocks: Iterable[Iterable[Outcome]], space: "FiniteProbabilitySpace") -> "Partition":
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
    def generated_by(cls, rv: "RandomVariable") -> "Partition":
        """Return the partition whose blocks are the level sets of ``rv``."""
        groups: dict[float, set[Outcome]] = {}
        for outcome, value in rv.values.items():
            groups.setdefault(float(value), set()).add(outcome)
        return cls.from_blocks(groups.values(), rv.space)

    def refines(self, coarser: "Partition") -> bool:
        """Return whether every block here lies inside a block of ``coarser``."""
        return all(any(block.issubset(parent) for parent in coarser.blocks) for block in self.blocks)


__all__ = ["Partition"]
