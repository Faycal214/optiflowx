"""Finite partitions used as sigma-fields in MSPRO Chapter 4."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Hashable

Outcome = Hashable


@dataclass(frozen=True)
class Partition:
    """A finite partition representing a finite sigma-field.
    
    Mathematical object
    ------------------
    Public stochastic object exposed by the StochX API.
    
    Course basis
    ------------
    The implementation follows the corresponding MSPRO course material documented by StochX.
    
    Parameters
    ----------
    blocks : tuple[frozenset[Outcome], ...]
        Partition blocks.
    
    Examples
    --------
    See the executable examples for `partition.py` and the API reference."""

    blocks: tuple[frozenset[Outcome], ...]

    @property
    def n_blocks(self) -> int:
        """Return the number of blocks in the partition.
        
        
        Returns
        -------
        int
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `n_blocks`."""
        return len(self.blocks)

    def contains(self, outcome: Outcome) -> int:
        """Return the index of the block containing ``outcome``.
        
        Parameters
        ----------
        outcome : Outcome
            Input argument.
        
        Returns
        -------
        int
            Result produced by the operation.
        
        Raises
        ------
        ValueError
            Raised when an input or mathematical precondition is violated.
        
        Examples
        --------
        See the executable examples and API reference for `contains`."""
        for index, block in enumerate(self.blocks):
            if outcome in block:
                return index
        raise ValueError(f"unknown outcome: {outcome!r}")

    @classmethod
    def from_blocks(cls, blocks: Iterable[Iterable[Outcome]], space: "FiniteProbabilitySpace") -> "Partition":
        """Build a partition after checking coverage and disjointness.
        
        Parameters
        ----------
        blocks : Iterable[Iterable[Outcome]]
            Partition blocks.
        space : 'FiniteProbabilitySpace'
            Input argument.
        
        Returns
        -------
        'Partition'
            Result produced by the operation.
        
        Raises
        ------
        ValueError
            Raised when an input or mathematical precondition is violated.
        
        Examples
        --------
        See the executable examples and API reference for `from_blocks`."""
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
        """Return the partition whose blocks are the level sets of ``rv``.
        
        Parameters
        ----------
        rv : 'RandomVariable'
            Input argument.
        
        Returns
        -------
        'Partition'
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `generated_by`."""
        groups: dict[float, set[Outcome]] = {}
        for outcome, value in rv.values.items():
            groups.setdefault(float(value), set()).add(outcome)
        return cls.from_blocks(groups.values(), rv.space)

    def refines(self, coarser: "Partition") -> bool:
        """Return whether every block here lies inside a block of ``coarser``.
        
        Parameters
        ----------
        coarser : 'Partition'
            Coarser partition.
        
        Returns
        -------
        bool
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `refines`."""
        return all(any(block.issubset(parent) for parent in coarser.blocks) for block in self.blocks)


__all__ = ["Partition"]
