"""Stopped processes from MSPRO Chapter 5."""

from __future__ import annotations

import numpy as np

from .random_variable import RandomVariable
from .stopping_time import StoppingTime


class StoppedProcess:
    """Stopped process ``X^T_n = X_{n wedge T}``.
    
    Mathematical object
    ------------------
    Public stochastic object exposed by the OptiFlowX API.
    
    Course basis
    ------------
    The implementation follows the corresponding MSPRO course material documented by OptiFlowX.
    
    Parameters
    ----------
    process
        Input argument.
    stopping_time : StoppingTime
        Input argument.
    
    Examples
    --------
    See the executable examples for `stopped_process.py` and the API reference."""

    def __init__(self, process, stopping_time: StoppingTime) -> None:
        """Public stochastic operation.
        
        Parameters
        ----------
        process
            Input argument.
        stopping_time : StoppingTime
            Input argument.
        
        Raises
        ------
        ValueError
            Raised when an input or mathematical precondition is violated.
        
        Examples
        --------
        See the executable examples and API reference for `__init__`."""
        if not process or process[0].space is not stopping_time.space:
            raise ValueError("process and stopping time must share a probability space")
        self.process = tuple(process)
        self.stopping_time = stopping_time

    def values(self, n: int) -> RandomVariable:
        """Return the stopped random variable ``X^T_n``.
        
        Parameters
        ----------
        n : int
            Integer index required by the operation.
        
        Returns
        -------
        RandomVariable
            Result produced by the operation.
        
        Raises
        ------
        ValueError
            Raised when an input or mathematical precondition is violated.
        
        Examples
        --------
        See the executable examples and API reference for `values`."""
        if n < 0 or n >= len(self.process):
            raise ValueError("n is outside the supplied process")
        values = {}
        for outcome in self.process[0].space.outcomes:
            tau = self.stopping_time.values[outcome]
            index = n if np.isinf(tau) else min(n, int(tau))
            values[outcome] = self.process[index].values[outcome]
        return RandomVariable(self.process[0].space, values, name=f"X^T_{n}")

    def sequence(self):
        """Return the complete supplied sequence of stopped variables.
        
        
        Examples
        --------
        See the executable examples and API reference for `sequence`."""
        return tuple(self.values(n) for n in range(len(self.process)))

    def terminal_value(self):
        """Return ``X_T`` when the stopping time is finite almost surely.
        
        
        Raises
        ------
        ValueError
            Raised when an input or mathematical precondition is violated.
        
        Examples
        --------
        See the executable examples and API reference for `terminal_value`."""
        if any(np.isinf(v) for v in self.stopping_time.values.values()):
            raise ValueError("X_T is undefined when T can be infinity")
        values = {
            o: self.process[int(self.stopping_time.values[o])].values[o]
            for o in self.process[0].space.outcomes
        }
        return RandomVariable(self.process[0].space, values, name="X_T")


__all__ = ["StoppedProcess"]
