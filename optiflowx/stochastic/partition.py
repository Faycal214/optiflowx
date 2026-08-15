"""Public API for finite partitions used as sigma-fields in Chapter 4."""

from .conditional import Partition as _Partition


class Partition(_Partition):
    """Finite partition representing the information carried by a sigma-field."""

    @property
    def n_blocks(self) -> int:
        """Return the number of blocks in the partition."""
        return len(self.blocks)

    def contains(self, outcome) -> int:
        """Return the index of the block containing ``outcome``."""
        for index, block in enumerate(self.blocks):
            if outcome in block:
                return index
        raise ValueError(f"unknown outcome: {outcome!r}")


__all__ = ["Partition"]
