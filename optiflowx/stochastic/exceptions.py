"""Exceptions raised by OptiFlowX stochastic models."""


class OptiFlowXError(Exception):
    """Base exception for OptiFlowX-specific failures."""


class ValidationError(OptiFlowXError, ValueError):
    """Base exception for invalid mathematical model inputs."""


class MatrixValidationError(ValidationError):
    """Raised when a matrix violates its required structural constraints."""


class ProbabilityValidationError(ValidationError):
    """Raised when probabilities do not define a valid probability law."""


class GeneratorValidationError(MatrixValidationError):
    """Raised when a CTMC generator does not satisfy generator conditions."""


class NumericalError(OptiFlowXError):
    """Raised when a numerical algorithm cannot safely complete."""


__all__ = [
    "OptiFlowXError",
    "ValidationError",
    "MatrixValidationError",
    "ProbabilityValidationError",
    "GeneratorValidationError",
    "NumericalError",
]
