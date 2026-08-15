"""Public API for finite homogeneous continuous-time Markov chains (CMTC)."""

from __future__ import annotations

from typing import Hashable

from .cmtc import CTMCPath, ContinuousTimeMarkovChain as _ContinuousTimeMarkovChain

State = Hashable


class ContinuousTimeMarkovChain(_ContinuousTimeMarkovChain):
    """Finite-state homogeneous continuous-time Markov chain.

    The generator ``Q`` is the primary mathematical object. Transition
    matrices satisfy ``P(t)=exp(tQ)`` in the finite-state setting of the course.
    """

    @property
    def generator(self):
        """Return the infinitesimal generator ``Q``."""
        return self.generator_matrix

    def transition_matrix_at(self, t: float):
        """Return ``P(t)=exp(tQ)``."""
        return self.transition_matrix(t)

    def forward_equation(self, t: float):
        """Return the forward Kolmogorov derivative ``P(t)Q``."""
        return self.forward_derivative(t)

    def backward_equation(self, t: float):
        """Return the backward Kolmogorov derivative ``QP(t)``."""
        return self.backward_derivative(t)

    def jump_chain(self):
        """Return the embedded discrete-time Markov chain as a public ``MarkovChain``."""
        from .markov_chain import MarkovChain

        return MarkovChain(self.jump_chain_matrix(), self.states, tolerance=self._tolerance)


__all__ = ["CTMCPath", "ContinuousTimeMarkovChain"]
