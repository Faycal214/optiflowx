import numpy as np
import pytest

from stochx.stochastic.exceptions import (
    GeneratorValidationError,
    MatrixValidationError,
    ProbabilityValidationError,
)
from stochx.stochastic.validation import (
    DEFAULT_TOLERANCE,
    normalize_stochastic_matrix,
    validate_generator,
    validate_probability_vector,
    validate_stochastic_matrix,
    validate_tolerance,
)


def test_default_tolerance_and_validation():
    assert DEFAULT_TOLERANCE == pytest.approx(1e-12)
    assert validate_tolerance(DEFAULT_TOLERANCE) == pytest.approx(DEFAULT_TOLERANCE)
    with pytest.raises(ValueError):
        validate_tolerance(0.0)
    with pytest.raises(ValueError):
        validate_tolerance(float("nan"))


def test_stochastic_matrix_validation_and_normalization():
    matrix = np.array([[0.2, 0.8 + 1e-13], [0.4, 0.6 - 1e-13]])
    checked = validate_stochastic_matrix(matrix)
    np.testing.assert_allclose(checked.sum(axis=1), [1.0, 1.0])
    np.testing.assert_array_less(-checked, np.full_like(checked, 1e-12))

    with pytest.raises(MatrixValidationError):
        validate_stochastic_matrix([[0.2, 0.7], [0.4, 0.6]])
    with pytest.raises(MatrixValidationError):
        validate_stochastic_matrix([[1.1, -0.1], [0.0, 1.0]])

    drifted = np.array([[0.5000000000001, 0.4999999999998], [0.2, 0.8000000000002]])
    normalized = normalize_stochastic_matrix(drifted)
    np.testing.assert_allclose(normalized.sum(axis=1), 1.0)


def test_probability_vector_validation():
    vector = validate_probability_vector([0.25, 0.75 + 1e-13], 2)
    np.testing.assert_allclose(vector.sum(), 1.0)
    with pytest.raises(ProbabilityValidationError):
        validate_probability_vector([0.2, 0.7], 2)
    with pytest.raises(ProbabilityValidationError):
        validate_probability_vector([0.2, -0.2], 2)


def test_generator_validation():
    q = validate_generator([[-2.0, 2.0], [1.0, -1.0]])
    np.testing.assert_allclose(q.sum(axis=1), 0.0)
    assert np.all(np.diag(q) <= 0.0)
    assert q[0, 1] >= 0.0 and q[1, 0] >= 0.0

    with pytest.raises(GeneratorValidationError):
        validate_generator([[1.0, -1.0], [0.0, 0.0]])
    with pytest.raises(GeneratorValidationError):
        validate_generator([[-1.0, 0.5], [0.0, 0.0]])
