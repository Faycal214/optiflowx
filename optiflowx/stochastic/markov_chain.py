"""Public API for finite homogeneous discrete-time Markov chains."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Hashable

from .markov import MarkovChain as _MarkovChain

State = Hashable


class MarkovChain(_MarkovChain):
    """Finite-state homogeneous discrete-time Markov chain.

    Parameters
    ----------
    transition_matrix:
        Square row-stochastic matrix ``P``.
    states:
        Optional state labels. Defaults to ``0, ..., n-1``.
    tolerance:
        Numerical tolerance used for validation and graph decisions.
    """

    def transition_matrix_at(self, n: int):
        """Return the ``n``-step transition matrix ``P^n``."""
        return self.n_step_transition(n)

    def first_passage_probability(self, source: State, target: State, n: int) -> float:
        """Return ``P_source(T_target = n)``."""
        return self.first_visit_probability(source, target, n)

    def mean_hitting_time(self, source: State, target: State) -> float:
        """Return the expected first hitting time of ``target`` from ``source``."""
        return self.expected_hitting_time(source, target)

    def jump_chain(self):
        """Return the chain itself; discrete-time chains already are the embedded chain."""
        return self


__all__ = ["MarkovChain"]
