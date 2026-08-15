"""Exceptions raised by StochX stochastic models."""


class StochXError(Exception):
    """Base exception for StochX-specific failures."""


class ValidationError(StochXError, ValueError):
    """Base exception for invalid mathematical model inputs."""


class MatrixValidationError(ValidationError):
    """Raised when a matrix violates its required structural constraints."""


class ProbabilityValidationError(ValidationError):
    """Raised when probabilities do not define a valid probability law."""


class GeneratorValidationError(MatrixValidationError):
    """Raised when a CTMC generator does not satisfy generator conditions."""


class NumericalError(StochXError):
    """Raised when a numerical algorithm cannot safely complete."""


__all__ = [
    "StochXError",
    "ValidationError",
    "MatrixValidationError",
    "ProbabilityValidationError",
    "GeneratorValidationError",
    "NumericalError",
]
