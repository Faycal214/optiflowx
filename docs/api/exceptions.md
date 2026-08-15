# Exceptions

OptiFlowX exposes semantic exceptions for invalid stochastic objects and numerical failures. They remain ordinary Python exception subclasses, so callers can still handle the corresponding standard exception categories when appropriate.

## Exception hierarchy

```text
OptiFlowXError
├── ValidationError
│   ├── MatrixValidationError
│   ├── ProbabilityValidationError
│   └── GeneratorValidationError
└── NumericalError
```

## ValidationError

Base class for invalid stochastic-object definitions.

## MatrixValidationError

Raised when a transition matrix or related matrix violates the package's matrix contract.

Typical examples include non-square matrices, non-finite entries, invalid row sums, or invalid probability values beyond the configured tolerance.

## ProbabilityValidationError

Raised when a probability, probability vector, or finite probability-space law is invalid.

## GeneratorValidationError

Raised when a CTMC generator violates the infinitesimal-generator conditions.

For off-diagonal entries and row sums:

$$q_{ij}\ge0\;(i\ne j),\qquad\sum_jq_{ij}=0.$$

## NumericalError

Raised when a numerical routine cannot produce a valid result under its configured numerical contract.
