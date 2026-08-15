"""Discrete-time martingales from MSPRO Chapter 5."""

from __future__ import annotations

import numpy as np

from .filtration import Filtration
from .random_variable import RandomVariable
from .stopping_time import StoppingTime
from .stopped_process import StoppedProcess


class Martingale:
    """Discrete-time adapted process tested against martingale identities."""

    def __init__(self, process, filtration: Filtration) -> None:
        if not process or len(process) > len(filtration):
            raise ValueError("process and filtration lengths are incompatible")
        space = process[0].space
        if any(rv.space is not space for rv in process):
            raise ValueError("all process variables must belong to the same space")
        if not filtration.is_adapted(process):
            raise ValueError("process must be adapted to the filtration")
        self.process = tuple(process)
        self.filtration = filtration

    @property
    def n_steps(self) -> int:
        """Return the number of supplied process variables."""
        return len(self.process)

    def value_at(self, n: int) -> RandomVariable:
        """Return ``X_n``."""
        if n < 0 or n >= len(self.process):
            raise ValueError("n is outside the supplied process")
        return self.process[n]

    def conditional_next(self, n: int) -> RandomVariable:
        """Return ``E[X_{n+1}|F_n]``."""
        if n < 0 or n + 1 >= len(self.process):
            raise ValueError("n must have a next process value")
        return self.process[n].space.conditional_expectation(self.process[n + 1], self.filtration[n])

    def martingale_residual(self, n: int) -> RandomVariable:
        """Return ``E[X_{n+1}|F_n]-X_n``."""
        return self.conditional_next(n) - self.process[n]

    def is_martingale(self, *, atol: float = 1e-10) -> bool:
        """Return whether the process satisfies the martingale equality."""
        return all(np.max(np.abs(self.martingale_residual(n).array())) <= atol for n in range(len(self.process) - 1))

    def is_submartingale(self, *, atol: float = 1e-10) -> bool:
        """Return whether the process satisfies the submartingale inequality."""
        return all(np.min(self.martingale_residual(n).array()) >= -atol for n in range(len(self.process) - 1))

    def is_supermartingale(self, *, atol: float = 1e-10) -> bool:
        """Return whether the process satisfies the supermartingale inequality."""
        return all(np.max(self.martingale_residual(n).array()) <= atol for n in range(len(self.process) - 1))

    def expectations(self) -> np.ndarray:
        """Return the sequence ``(E[X_n])``."""
        return np.asarray([rv.expectation() for rv in self.process])

    def conditional_future(self, n: int, k: int) -> RandomVariable:
        """Return ``E[X_{n+k}|F_n]``."""
        if n < 0 or k < 0 or n + k >= len(self.process):
            raise ValueError("invalid time indices")
        return self.process[n].space.conditional_expectation(self.process[n + k], self.filtration[n])

    def transform(self, function, *, name: str | None = None) -> "Martingale":
        """Apply a scalar transformation pointwise to the supplied process."""
        transformed = tuple(rv.apply(function, name=name) for rv in self.process)
        return Martingale(transformed, self.filtration)

    def stopped(self, stopping_time: StoppingTime) -> StoppedProcess:
        """Construct the stopped process ``X_{n wedge T}``."""
        return StoppedProcess(self.process, stopping_time)

    @classmethod
    def doob(cls, terminal: RandomVariable, filtration: Filtration) -> "Martingale":
        """Construct the Doob martingale ``M_n=E[X|F_n]``."""
        process = tuple(terminal.space.conditional_expectation(terminal, partition) for partition in filtration.partitions)
        return cls(process, filtration)


__all__ = ["Martingale"]
