"""Public API for birth-death processes.

Birth-death processes are a special class of continuous-time Markov chains in
MSPRO Chapter 3, with rates lambda_k for births and mu_k for deaths.
"""

from .birth_death import BirthDeathProcess as _BirthDeathProcess


class BirthDeathProcess(_BirthDeathProcess):
    """Birth-death process with state-dependent birth and death rates."""

    @property
    def generator(self):
        """Return the finite generator matrix ``Q`` when ``max_state`` is set."""
        return self.generator_matrix()

    def jump_chain(self):
        """Return the embedded discrete-time transition matrix."""
        return self.jump_chain_matrix()


__all__ = ["BirthDeathProcess"]
