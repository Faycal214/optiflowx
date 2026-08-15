"""Stopping times from MSPRO Chapter 5."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .filtration import Filtration
from .probability_space import FiniteProbabilitySpace


@dataclass(frozen=True)
class StoppingTime:
    """Stopping time with values in ``N union {infinity}``.
    
    Mathematical object
    ------------------
    Public stochastic object exposed by the OptiFlowX API.
    
    Course basis
    ------------
    The implementation follows the corresponding MSPRO course material documented by OptiFlowX.
    
    Parameters
    ----------
    space : FiniteProbabilitySpace
        Input argument.
    values : dict
        Values or mapping.
    filtration : Filtration
        Input argument.
    
    Examples
    --------
    See the executable examples for `stopping_time.py` and the API reference."""

    space: FiniteProbabilitySpace
    values: dict
    filtration: Filtration

    def __post_init__(self) -> None:
        if set(self.values) != set(self.space.outcomes):
            raise ValueError("stopping-time values must cover all outcomes")
        for value in self.values.values():
            if np.isinf(value):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError("stopping times must be non-negative integers or infinity")
        for n, partition in enumerate(self.filtration.partitions):
            event = {o for o in self.space.outcomes if self.values[o] == n}
            if not all(block <= event or block.isdisjoint(event) for block in partition.blocks):
                raise ValueError(f"{{T={n}}} is not in F_{n}")

    @classmethod
    def from_values(cls, space, values, filtration):
        """Create a stopping time from outcome-wise values.
        
        Parameters
        ----------
        space
            Input argument.
        values
            Values or mapping.
        filtration
            Input argument.
        
        Examples
        --------
        See the executable examples and API reference for `from_values`."""
        return cls(space, dict(values), filtration)

    def minimum(self, other):
        """Return ``T wedge S``.
        
        Parameters
        ----------
        other
            Input argument.
        
        Examples
        --------
        See the executable examples and API reference for `minimum`."""
        self._same(other)
        return StoppingTime(self.space, {o: min(self.values[o], other.values[o]) for o in self.space.outcomes}, self.filtration)

    def maximum(self, other):
        """Return ``T vee S``.
        
        Parameters
        ----------
        other
            Input argument.
        
        Examples
        --------
        See the executable examples and API reference for `maximum`."""
        self._same(other)
        return StoppingTime(self.space, {o: max(self.values[o], other.values[o]) for o in self.space.outcomes}, self.filtration)

    def add(self, other):
        """Return ``T+S``.
        
        Parameters
        ----------
        other
            Input argument.
        
        Examples
        --------
        See the executable examples and API reference for `add`."""
        self._same(other)
        values = {
            o: np.inf if np.isinf(self.values[o]) or np.isinf(other.values[o]) else self.values[o] + other.values[o]
            for o in self.space.outcomes
        }
        return StoppingTime(self.space, values, self.filtration)

    def _same(self, other) -> None:
        if other.space is not self.space or other.filtration.partitions != self.filtration.partitions:
            raise ValueError("stopping times must share the same space and filtration")


__all__ = ["StoppingTime"]
