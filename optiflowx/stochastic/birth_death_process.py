"""Public API for birth-death processes.

Birth-death processes are treated as a specialized class of continuous-time
Markov chains in MSPRO Chapter 3.
"""

from .birth_death import BirthDeathProcess

__all__ = ["BirthDeathProcess"]
