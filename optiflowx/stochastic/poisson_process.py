"""Public API for homogeneous and non-homogeneous Poisson processes.

The mathematical implementation remains in :mod:`poisson` during the API
migration.  The module establishes the canonical public import path.
"""

from .poisson import NonHomogeneousPoissonProcess, PoissonProcess

__all__ = ["PoissonProcess", "NonHomogeneousPoissonProcess"]
