"""Internal implementation for discrete-time martingales.

The public API is exposed from ``optiflowx.stochastic.martingale``.  This
module preserves the computational implementation while the public facade is
standardized.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .conditional import FiniteProbabilitySpace, Partition, RandomVariable


class Filtration:
    """Increasing sequence of finite sigma-fields represented by partitions."""

    def __init__(self, partitions: Sequence[Partition]) -> None:
        if not partitions:
            raise ValueError("a filtration must contain at least one partition")
        self.partitions = tuple(partitions)
        for previous, current in zip(self.partitions, self.partitions[1:]):
            if not current.refines(previous):
                raise ValueError("filtration partitions must be increasing")

    def __getitem__(self, n: int) -> Partition:
        return self.partitions[n]

    def __len__(self) -> int:
        return len(self.partitions)

    @classmethod
    def natural(cls, process: Sequence[RandomVariable]) -> "Filtration":
        if not process:
            raise ValueError("process must not be empty")
        space = process[0].space
        if any(rv.space is not space for rv in process):
            raise ValueError("all variables must belong to the same space")
        partitions = []
        for n in range(len(process)):
            groups = {}
            for outcome in space.outcomes:
                key = tuple(process[k].values[outcome] for k in range(n + 1))
                groups.setdefault(key, set()).add(outcome)
            partitions.append(Partition.from_blocks(groups.values(), space))
        return cls(partitions)

    def is_adapted(self, process: Sequence[RandomVariable]) -> bool:
        if len(process) > len(self):
            return False
        return all(
            process[n].space.check_measurable(process[n], self[n])
            for n in range(len(process))
        )


@dataclass(frozen=True)
class StoppingTime:
    """Internal stopping-time implementation."""

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
        return cls(space, dict(values), filtration)

    def minimum(self, other):
        self._same(other)
        return StoppingTime(
            self.space,
            {o: min(self.values[o], other.values[o]) for o in self.space.outcomes},
            self.filtration,
        )

    def maximum(self, other):
        self._same(other)
        return StoppingTime(
            self.space,
            {o: max(self.values[o], other.values[o]) for o in self.space.outcomes},
            self.filtration,
        )

    def add(self, other):
        self._same(other)
        vals = {
            o: (
                np.inf
                if np.isinf(self.values[o]) or np.isinf(other.values[o])
                else self.values[o] + other.values[o]
            )
            for o in self.space.outcomes
        }
        return StoppingTime(self.space, vals, self.filtration)

    def _same(self, other) -> None:
        if other.space is not self.space or other.filtration.partitions != self.filtration.partitions:
            raise ValueError("stopping times must share the same space and filtration")


class Martingale:
    """Internal discrete-time martingale implementation."""

    def __init__(self, process: Sequence[RandomVariable], filtration: Filtration) -> None:
        if not process or len(process) > len(filtration):
            raise ValueError("process and filtration lengths are incompatible")
        space = process[0].space
        if any(rv.space is not space for rv in process):
            raise ValueError("all process variables must belong to the same space")
        if not filtration.is_adapted(process):
            raise ValueError("process must be adapted to the filtration")
        self.process = tuple(process)
        self.filtration = filtration

    def conditional_next(self, n: int) -> RandomVariable:
        if n < 0 or n + 1 >= len(self.process):
            raise ValueError("n must have a next process value")
        return self.process[n].space.conditional_expectation(
            self.process[n + 1], self.filtration[n]
        )

    def martingale_residual(self, n: int) -> RandomVariable:
        return self.conditional_next(n) - self.process[n]

    def is_martingale(self, *, atol: float = 1e-10) -> bool:
        return all(
            np.max(np.abs(self.martingale_residual(n).array())) <= atol
            for n in range(len(self.process) - 1)
        )

    def is_submartingale(self, *, atol: float = 1e-10) -> bool:
        return all(
            np.min(self.martingale_residual(n).array()) >= -atol
            for n in range(len(self.process) - 1)
        )

    def is_supermartingale(self, *, atol: float = 1e-10) -> bool:
        return all(
            np.max(self.martingale_residual(n).array()) <= atol
            for n in range(len(self.process) - 1)
        )

    def expectations(self) -> np.ndarray:
        return np.asarray([rv.expectation() for rv in self.process])

    def conditional_future(self, n: int, k: int) -> RandomVariable:
        if n < 0 or k < 0 or n + k >= len(self.process):
            raise ValueError("invalid time indices")
        return self.process[n].space.conditional_expectation(
            self.process[n + k], self.filtration[n]
        )

    def stopped(self, stopping_time: StoppingTime):
        return StoppedProcess(self.process, stopping_time)

    @classmethod
    def doob(cls, terminal: RandomVariable, filtration: Filtration):
        process = tuple(
            terminal.space.conditional_expectation(terminal, partition)
            for partition in filtration.partitions
        )
        return cls(process, filtration)


class StoppedProcess:
    """Internal stopped-process implementation."""

    def __init__(self, process: Sequence[RandomVariable], stopping_time: StoppingTime) -> None:
        if not process or process[0].space is not stopping_time.space:
            raise ValueError("process and stopping time must share a probability space")
        self.process = tuple(process)
        self.stopping_time = stopping_time

    def values(self, n: int) -> RandomVariable:
        if n < 0 or n >= len(self.process):
            raise ValueError("n is outside the supplied process")
        values = {}
        for outcome in self.process[0].space.outcomes:
            tau = self.stopping_time.values[outcome]
            index = n if np.isinf(tau) else min(n, int(tau))
            values[outcome] = self.process[index].values[outcome]
        return RandomVariable(self.process[0].space, values, name=f"X^T_{n}")

    def sequence(self):
        return tuple(self.values(n) for n in range(len(self.process)))

    def terminal_value(self):
        if any(np.isinf(v) for v in self.stopping_time.values.values()):
            raise ValueError("X_T is undefined when T can be infinity")
        values = {
            o: self.process[int(self.stopping_time.values[o])].values[o]
            for o in self.process[0].space.outcomes
        }
        return RandomVariable(self.process[0].space, values, name="X_T")
