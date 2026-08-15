"""Public API for birth-death processes."""

from .birth_death import BirthDeathProcess as _BirthDeathProcess


class BirthDeathProcess(_BirthDeathProcess):
    """Birth-death process with state-dependent birth and death rates.

    In Chapter 3 the only possible jumps are ``k -> k+1`` with rate
    ``lambda_k`` and ``k -> k-1`` with rate ``mu_k``.
    """

    @property
    def generator(self):
        """Return the finite generator ``Q`` when ``max_state`` is specified."""
        return self.generator_matrix()

    def jump_chain(self):
        """Return the embedded jump chain as a public ``MarkovChain``."""
        from .markov_chain import MarkovChain

        return MarkovChain(self.jump_chain_matrix())

    def to_ctmc(self):
        """Return the corresponding public ``ContinuousTimeMarkovChain``."""
        from .continuous_time_markov_chain import ContinuousTimeMarkovChain

        return ContinuousTimeMarkovChain(self.generator_matrix())

    def stationary_weights_at(self, n_terms: int):
        """Return the stationary product weights through state ``n_terms``."""
        return self.stationary_weights(n_terms)


__all__ = ["BirthDeathProcess"]
