"""Public API for finite homogeneous discrete-time Markov chains.

The implementation remains in :mod:`optiflowx.stochastic.markov` during the
API migration.  This module establishes the canonical import path and adds
library-style method names while preserving the existing mathematical model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Hashable

from .markov import MarkovChain as _MarkovChain

State = Hashable


class MarkovChain(_MarkovChain):
    """Finite-state homogeneous discrete-time Markov chain.

    The chain is represented by a stochastic transition matrix ``P``.  The
    course relation ``P^(n) = P**n`` is exposed as ``transition_matrix_at``.

    Parameters
    ----------
    transition_matrix:
        Square row-stochastic transition matrix.
    states:
        Optional state labels. Defaults to ``0, ..., n-1``.
    tolerance:
        Numerical tolerance used by the underlying implementation.
    """

    def transition_matrix_at(self, n: int):
        """Return the ``n``-step transition matrix ``P^n``."""
        return self.n_step_transition(n)

    def first_passage_probability(self, source: State, target: State, n: int) -> float:
        """Return the probability ``P_i(T_j = n)`` of first hitting ``j`` at ``n``."""
        return self.first_visit_probability(source, target, n)

    def mean_hitting_time(self, source: State, target: State) -> float:
        """Return the expected first hitting time of ``target`` from ``source``."""
        return self.expected_hitting_time(source, target)
