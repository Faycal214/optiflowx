"""Generic trajectory-analysis helpers."""

from collections.abc import Sequence
from typing import Hashable

import numpy as np

State = Hashable


def empirical_state_frequencies(path: Sequence[State], states: Sequence[State]) -> np.ndarray:
    """Return empirical visit frequencies of states along a discrete trajectory."""
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
