"""Public API for discrete-time martingales from MSPRO Chapter 5.

The canonical public classes live here; mathematical validation and finite
conditional-expectation calculations remain delegated to the existing core.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .martingale import Filtration as _Filtration
from .martingale import Martingale as _Martingale
from .martingale import StoppedProcess as _StoppedProcess
from .martingale import StoppingTime as _StoppingTime


class Filtration(_Filtration):
    """Increasing finite filtration represented by partitions."""

    @property
    def n_steps(self) -> int:
        """Return the number of filtration levels."""
        return len(self)

    def at(self, n: int):
        """Return ``F_n``."""
        return self[n]


class StoppingTime(_StoppingTime):
    """Discrete stopping time with values in ``N union {infinity}``."""

    @classmethod
    def from_values(cls, space, values, filtration):
        """Create a stopping time from outcome-wise values."""
        return cls(space, dict(values), filtration)

    def minimum(self, other):
        """Return ``T wedge S`` as a canonical ``StoppingTime``."""
        self._same(other)
        return StoppingTime(
            self.space,
            {o: min(self.values[o], other.values[o]) for o in self.space.outcomes},
            self.filtration,
        )

    def maximum(self, other):
        """Return ``T vee S`` as a canonical ``StoppingTime``."""
        self._same(other)
        return StoppingTime(
            self.space,
            {o: max(self.values[o], other.values[o]) for o in self.space.outcomes},
            self.filtration,
        )

    def add(self, other):
        """Return ``T+S`` as a canonical ``StoppingTime``."""
        self._same(other)
        values = {}
        for outcome in self.space.outcomes:
            left, right = self.values[outcome], other.values[outcome]
            values[outcome] = np.inf if np.isinf(left) or np.isinf(right) else left + right
        return StoppingTime(self.space, values, self.filtration)


class StoppedProcess(_StoppedProcess):
    """Stopped process ``X^T_n = X_{n wedge T}``."""

    def values(self, n: int):
        """Return the stopped random variable ``X^T_n``."""
        from .random_variable import RandomVariable

        if n < 0 or n >= len(self.process):
            raise ValueError("n is outside the supplied process")
        values = {}
        for outcome in self.process[0].space.outcomes:
            tau = self.stopping_time.values[outcome]
            index = n if np.isinf(tau) else min(n, int(tau))
            values[outcome] = self.process[index].values[outcome]
        return RandomVariable(self.process[0].space, values, name=f"X^T_{n}")

    def sequence(self):
        """Return the complete supplied sequence of stopped variables."""
        return tuple(self.values(n) for n in range(len(self.process)))

    def terminal_value(self):
        """Return ``X_T`` when the stopping time is finite almost surely."""
        if any(np.isinf(v) for v in self.stopping_time.values.values()):
            raise ValueError("X_T is undefined when T can be infinity")
        from .random_variable import RandomVariable

        values = {
            o: self.process[int(self.stopping_time.values[o])].values[o]
            for o in self.process[0].space.outcomes
        }
        return RandomVariable(self.process[0].space, values, name="X_T")


class Martingale(_Martingale):
    """Discrete-time adapted process tested against the martingale identities."""

    def stopped(self, stopping_time: StoppingTime) -> StoppedProcess:
        """Construct the canonical stopped process ``X_{n wedge T}``."""
        return StoppedProcess(self.process, stopping_time)

    @classmethod
    def doob(cls, terminal, filtration: Filtration) -> "Martingale":
        """Construct the Doob martingale ``M_n = E[X | F_n]``."""
        process = tuple(
            terminal.space.conditional_expectation(terminal, partition)
            for partition in filtration.partitions
        )
        return cls(process, filtration)

    @property
    def n_steps(self) -> int:
        """Return the number of process variables supplied."""
        return len(self.process)

    def value_at(self, n: int):
        """Return ``X_n``."""
        return self.process[n]


__all__ = ["Filtration", "Martingale", "StoppingTime", "StoppedProcess"]
