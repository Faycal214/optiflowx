"""Generic trajectory-analysis helpers."""

from collections.abc import Sequence
from typing import Hashable

import numpy as np

State = Hashable


def empirical_state_frequencies(path: Sequence[State], states: Sequence[State]) -> np.ndarray:
    """Return empirical visit frequencies of states along a discrete trajectory.
    
    Parameters
    ----------
    path : Sequence[State]
        Input argument.
    states : Sequence[State]
        State labels in matrix order.
    
    Returns
    -------
    np.ndarray
        Result produced by the operation.
    
    Raises
    ------
    ValueError
        Raised when an input or mathematical precondition is violated.
    
    Examples
    --------
    See the executable examples and API reference for `empirical_state_frequencies`."""
    labels = tuple(states)
    if not path:
        raise ValueError("path must not be empty")
    index = {state: i for i, state in enumerate(labels)}
    if len(index) != len(labels):
        raise ValueError("states must be unique")
    counts = np.zeros(len(labels), dtype=float)
    for state in path:
        if state not in index:
            raise ValueError(f"unknown state in path: {state!r}")
        counts[index[state]] += 1.0
    return counts / len(path)


__all__ = ["empirical_state_frequencies"]
