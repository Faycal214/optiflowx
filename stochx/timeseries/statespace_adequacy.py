"""Stage 11.6 adequacy tests for Kalman standardized innovations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .diagnostics import TestResult, ljung_box
from .statespace_diagnostics import KalmanInnovationDiagnosticsResult


@dataclass(frozen=True)
class StateSpaceAdequacyResult:
    """Auditable per-dimension adequacy tests for state-space innovations."""

    whiteness: tuple[TestResult, ...]
    normality: tuple[TestResult, ...]
    mean_zero: tuple[TestResult, ...]
    alpha: float
    lags: int
    dimensions: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "whiteness", tuple(self.whiteness))
        object.__setattr__(self, "normality", tuple(self.normality))
        object.__setattr__(self, "mean_zero", tuple(self.mean_zero))
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "lags", int(self.lags))
        object.__setattr__(self, "dimensions", int(self.dimensions))
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie strictly between 0 and 1")
        if self.lags < 1:
            raise ValueError("lags must be a positive integer")
        for name in ("whiteness", "normality", "mean_zero"):
            values = getattr(self, name)
            if len(values) != self.dimensions:
                raise ValueError(f"{name} must contain one result per observation dimension")
            if not all(isinstance(value, TestResult) for value in values):
                raise TypeError(f"{name} must contain TestResult objects")

    @property
    def passed_whiteness(self) -> bool:
        """Whether no dimension rejects the innovation-whiteness null."""
        return all(not result.reject for result in self.whiteness)

    @property
    def passed_normality(self) -> bool:
        """Whether no dimension rejects the normality null."""
        return all(not result.reject for result in self.normality)

    @property
    def passed_mean_zero(self) -> bool:
        """Whether no dimension rejects the zero-mean null."""
        return all(not result.reject for result in self.mean_zero)

    @property
    def adequate(self) -> bool:
        """Overall convenience decision across all configured tests."""
        return self.passed_whiteness and self.passed_normality and self.passed_mean_zero


def _normality_test(values: np.ndarray, *, alpha: float, dimension: int) -> TestResult:
    if values.size < 3:
        return TestResult(
            f"State-space normality dim {dimension}",
            np.nan,
            np.nan,
            "Standardized innovations are normally distributed",
            "Standardized innovations are not normally distributed",
            alpha,
        )
    statistic, pvalue = stats.jarque_bera(values)
    return TestResult(
        f"State-space Jarque-Bera dim {dimension}",
        float(statistic),
        float(pvalue),
        "Standardized innovations are normally distributed",
        "Standardized innovations are not normally distributed",
        alpha,
    )


def _mean_zero(values: np.ndarray, *, alpha: float, dimension: int) -> TestResult:
    if values.size < 2:
        return TestResult(
            f"State-space mean dim {dimension}",
            np.nan,
            np.nan,
            "E(z_t)=0",
            "E(z_t) differs from zero",
            alpha,
        )
    statistic, pvalue = stats.ttest_1samp(values, 0.0)
    return TestResult(
        f"State-space mean dim {dimension}",
        float(statistic),
        float(pvalue),
        "E(z_t)=0",
        "E(z_t) differs from zero",
        alpha,
    )


def state_space_adequacy(
    diagnostics: KalmanInnovationDiagnosticsResult,
    *,
    lags: int = 12,
    alpha: float = 0.05,
) -> StateSpaceAdequacyResult:
    """Run whiteness, normality and zero-mean tests on standardized innovations.

    Tests are evaluated independently for each observation dimension. Missing
    innovations are ignored by the existing diagnostic semantics. The Ljung-
    Box degrees of freedom are not reduced because these are state-space
    innovation diagnostics rather than ARMA residual diagnostics.
    """
    if not isinstance(diagnostics, KalmanInnovationDiagnosticsResult):
        raise TypeError("diagnostics must be a KalmanInnovationDiagnosticsResult")
    if not isinstance(lags, int) or isinstance(lags, bool) or lags < 1:
        raise ValueError("lags must be a positive integer")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")

    whiteness: list[TestResult] = []
    normality: list[TestResult] = []
    mean_zero: list[TestResult] = []

    for j in range(diagnostics.n_obs):
        values = diagnostics.standardized_innovations[:, j]
        values = values[np.isfinite(values)]

        if values.size < 3 or values.size <= lags:
            whiteness.append(
                TestResult(
                    f"State-space Ljung-Box dim {j + 1}",
                    np.nan,
                    np.nan,
                    f"Standardized innovations are uncorrelated through lag {lags}",
                    f"Standardized innovations show autocorrelation through lag {lags}",
                    alpha,
                )
            )
        else:
            result = ljung_box(values, lags=lags, alpha=alpha)
            whiteness.append(
                TestResult(
                    f"State-space Ljung-Box dim {j + 1}",
                    result.statistic,
                    result.pvalue,
                    f"Standardized innovations are uncorrelated through lag {lags}",
                    f"Standardized innovations show autocorrelation through lag {lags}",
                    alpha,
                )
            )

        normality.append(_normality_test(values, alpha=alpha, dimension=j + 1))
        mean_zero.append(_mean_zero(values, alpha=alpha, dimension=j + 1))

    return StateSpaceAdequacyResult(
        whiteness=tuple(whiteness),
        normality=tuple(normality),
        mean_zero=tuple(mean_zero),
        alpha=alpha,
        lags=lags,
        dimensions=diagnostics.n_obs,
    )
