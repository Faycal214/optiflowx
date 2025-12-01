"""Utilities to sanitize optimizer output for serialization and display.

Mainly converts NumPy scalar types to native Python types and recursively
handles lists/dicts so results are safe to JSON-serialize and log.
"""
from __future__ import annotations

from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - best-effort
    np = None  # type: ignore


def _sanitize_value(v: Any) -> Any:
    """Convert a single value to a JSON/print-friendly native Python type.

    - NumPy scalars -> native using `.item()`
    - NumPy arrays -> convert to Python lists
    - Other values returned unchanged
    """
    if np is not None:
        # NumPy scalar like np.int64, np.float64, np.bool_
        if isinstance(v, np.generic):
            try:
                return v.item()
            except Exception:
                return v
        # NumPy arrays
        if isinstance(v, (np.ndarray,)):
            try:
                return v.tolist()
            except Exception:
                return v

    # No special handling required
    return v


def sanitize_params(obj: Any) -> Any:
    """Recursively sanitize dicts/lists/tuples of parameters.

    Returns a new structure with NumPy types converted to native Python types.
    """
    if isinstance(obj, dict):
        return {k: sanitize_params(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [sanitize_params(v) for v in obj]
        return tuple(t) if isinstance(obj, tuple) else t
    # Primitive or unknown types
    return _sanitize_value(obj)
