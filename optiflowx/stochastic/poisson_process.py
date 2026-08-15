"""Public API for homogeneous and non-homogeneous Poisson processes."""

from .poisson import NonHomogeneousPoissonProcess as _NonHomogeneousPoissonProcess
from .poisson import PoissonProcess as _PoissonProcess


class PoissonProcess(_PoissonProcess):
    """Homogeneous Poisson process with constant rate ``lambda``.

    The course characterizes the process through independent stationary
    increments and the Poisson count law with parameter ``lambda * t``.
    """

    @property
    def lambda_(self) -> float:
        """Return the process rate ``lambda`` using a Python-safe name."""
        return self.rate

    def count(self, t: float, *, rng=None) -> int:
        """Sample the count ``N(t)`` over ``[0,t]``."""
        return self.count_sample(t, rng=rng)


class NonHomogeneousPoissonProcess(_NonHomogeneousPoissonProcess):
    """Non-homogeneous Poisson process defined through ``lambda(t)``."""

    @property
    def intensity_function(self):
        """Return the callable defining the time-varying intensity."""
        return self.intensity


__all__ = ["PoissonProcess", "NonHomogeneousPoissonProcess"]
