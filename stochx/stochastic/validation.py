"""Central validation and tolerance helpers for stochastic models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Hashable

import numpy as np

from .exceptions import (
    GeneratorValidationError,
    MatrixValidationError,
    ProbabilityValidationError,
)

DEFAULT_TOLERANCE = 1e-12
State = Hashable


def validate_tolerance(tolerance: float) -> float:
    """Validate a finite probability-scale tolerance in ``(0, 1)``."""
    value = float(tolerance)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("tolerance must be finite and strictly between 0 and 1")
    return value


def as_finite_square_matrix(
    matrix: Sequence[Sequence[float]],
    *,
    name: str = "matrix",
) -> np.ndarray:
    """Convert a matrix-like object to a finite non-empty square array."""
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] == 0:
        raise MatrixValidationError(f"{name} must be a non-empty square matrix")
    if not np.all(np.isfinite(array)):
        raise MatrixValidationError(f"{name} must contain only finite values")
    return array.copy()


def validate_stochastic_matrix(
    matrix: Sequence[Sequence[float]],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    name: str = "transition_matrix",
) -> np.ndarray:
    """Validate a finite row-stochastic matrix and return a normalized copy."""
    tol = validate_tolerance(tolerance)
    array = as_finite_square_matrix(matrix, name=name)
    if np.any(array < -tol):
        raise MatrixValidationError(f"{name} must be non-negative")
    if not np.allclose(array.sum(axis=1), 1.0, atol=tol, rtol=0.0):
        raise MatrixValidationError(f"{name} rows must sum to 1 within tolerance")
    array[np.abs(array) < tol] = 0.0
    array[array < 0.0] = 0.0
    return normalize_stochastic_matrix(array, tolerance=tol)


def normalize_stochastic_matrix(
    matrix: Sequence[Sequence[float]],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """Clean floating-point drift and renormalize stochastic rows."""
    tol = validate_tolerance(tolerance)
    array = as_finite_square_matrix(matrix, name="stochastic matrix")
    if np.any(array < -tol):
        raise MatrixValidationError("stochastic matrix contains entries below tolerance")
    array[np.abs(array) < tol] = 0.0
    array[array < 0.0] = 0.0
    row_sums = array.sum(axis=1)
    if np.any(row_sums <= tol):
        raise MatrixValidationError("stochastic matrix contains a zero-mass row")
    array /= row_sums[:, None]
    return array


def validate_probability_vector(
    values: Sequence[float],
    size: int,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    name: str = "probabilities",
) -> np.ndarray:
    """Validate a non-negative probability vector of a prescribed size."""
    tol = validate_tolerance(tolerance)
    array = np.asarray(values, dtype=float)
    if array.shape != (size,):
        raise ProbabilityValidationError(f"{name} must have shape ({size},)")
    if not np.all(np.isfinite(array)) or np.any(array < -tol):
        raise ProbabilityValidationError(f"{name} must be finite and non-negative")
    if not np.isclose(array.sum(), 1.0, atol=tol, rtol=0.0):
        raise ProbabilityValidationError(f"{name} must sum to 1 within tolerance")
    array = array.copy()
    array[np.abs(array) < tol] = 0.0
    array[array < 0.0] = 0.0
    return array / array.sum()


def validate_generator(
    generator: Sequence[Sequence[float]],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    name: str = "generator",
) -> np.ndarray:
    """Validate a finite CTMC generator matrix and return a copy.

    A valid finite-state generator has non-negative off-diagonal rates,
    non-positive diagonal entries, and every row summing to zero.
    """
    tol = validate_tolerance(tolerance)
    array = as_finite_square_matrix(generator, name=name)
    off_diagonal = array - np.diag(np.diag(array))
    if np.any(off_diagonal < -tol):
        raise GeneratorValidationError(f"{name} off-diagonal entries must be non-negative")
    if np.any(np.diag(array) > tol):
        raise GeneratorValidationError(f"{name} diagonal entries must be non-positive")
    if not np.allclose(array.sum(axis=1), 0.0, atol=tol, rtol=0.0):
        raise GeneratorValidationError(f"{name} rows must sum to zero within tolerance")
    array[np.abs(array) < tol] = 0.0
    off_diag_sum = array.sum(axis=1) - np.diag(array)
    array[np.diag_indices_from(array)] = -off_diag_sum
    return array


def validate_states(states: Sequence[State] | None, size: int) -> tuple[State, ...]:
    """Validate optional state labels and return a canonical tuple."""
    labels = tuple(range(size)) if states is None else tuple(states)
    if len(labels) != size or len(set(labels)) != size:
        raise ValueError("states must be unique and match matrix size")
    return labels


__all__ = [
    "DEFAULT_TOLERANCE",
    "as_finite_square_matrix",
    "normalize_stochastic_matrix",
    "validate_generator",
    "validate_probability_vector",
    "validate_states",
    "validate_stochastic_matrix",
    "validate_tolerance",
]
