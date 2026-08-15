"""Public API for discrete-time filtrations from MSPRO Chapter 5."""

from .martingale import Filtration as _Filtration


class Filtration(_Filtration):
    """Increasing sequence of finite sigma-fields represented by partitions."""

    @property
    def n_steps(self) -> int:
        """Return the number of filtration levels currently represented."""
        return len(self)

    def at(self, n: int):
        """Return the sigma-field/partition ``F_n``."""
        return self[n]


__all__ = ["Filtration"]
