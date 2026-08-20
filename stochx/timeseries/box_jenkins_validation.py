"""Stage 9.4 Box-Jenkins residual validation and adequacy filtering.

This layer composes the existing Stage 8 residual correlogram and the existing
StochX residual diagnostics. It never changes their numerical contracts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .box_jenkins_estimation import BoxJenkinsEstimationResult, EstimatedCandidate
from .diagnostics import (
    TestResult,
    arch_test,
    jarque_bera,
    mean_zero_test,
    normality_ks,
    residual_correlogram,
)


@dataclass(frozen=True)
class CandidateValidation:
    """Auditable validation decision for one estimated candidate."""

    order: tuple[int, int, int]
    estimation_success: bool
    serially_adequate: bool
    adequate: bool
    residual_nobs: int
    validation_lags: int
    alpha: float
    residual_correlogram: object | None
    mean_test: TestResult | None
    normality_test: TestResult | None
    ks_test: TestResult | None
    arch_test: TestResult | None
    required_lag_pvalues: tuple[float, ...]
    failed_checks: tuple[str, ...]
    rationale: str
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "order", tuple(int(v) for v in self.order))
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "residual_nobs", int(self.residual_nobs))
        object.__setattr__(self, "validation_lags", int(self.validation_lags))
        object.__setattr__(self, "required_lag_pvalues", tuple(float(v) for v in self.required_lag_pvalues))
        object.__setattr__(self, "failed_checks", tuple(str(v) for v in self.failed_checks))
        if self.error is not None:
            object.__setattr__(self, "error", str(self.error))

    @property
    def eligible(self) -> bool:
        """Whether the candidate may enter model selection."""
        return bool(self.estimation_success and self.adequate)


@dataclass(frozen=True)
class BoxJenkinsValidationResult:
    """Immutable validation results for the complete candidate set."""

    candidates: tuple[CandidateValidation, ...]

    @property
    def adequate(self) -> tuple[CandidateValidation, ...]:
        return tuple(c for c in self.candidates if c.eligible)

    @property
    def inadequate(self) -> tuple[CandidateValidation, ...]:
        return tuple(c for c in self.candidates if not c.eligible)

    @property
    def has_adequate_model(self) -> bool:
        return bool(self.adequate)

    def table(self) -> pd.DataFrame:
        rows = []
        for candidate in self.candidates:
            rows.append(
                {
                    "p": candidate.order[0],
                    "d": candidate.order[1],
                    "q": candidate.order[2],
                    "estimated": candidate.estimation_success,
                    "serially_adequate": candidate.serially_adequate,
                    "adequate": candidate.adequate,
                    "nobs": candidate.residual_nobs,
                    "validation_lags": candidate.validation_lags,
                    "failed_checks": "; ".join(candidate.failed_checks),
                    "error": candidate.error,
                    "rationale": candidate.rationale,
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "p", "d", "q", "estimated", "serially_adequate", "adequate",
                "nobs", "validation_lags", "failed_checks", "error", "rationale",
            ],
        )


def _required_pvalues(correlogram_result) -> tuple[float, ...]:
    """Return finite Ljung-Box probabilities only for lags with positive DF."""
    return tuple(float(p) for p, df in zip(correlogram_result.Prob, correlogram_result.DF) if int(df) > 0 and np.isfinite(p))


def _validate_candidate(
    candidate: EstimatedCandidate,
    *,
    lags: int,
    alpha: float,
    require_mean_zero: bool,
    require_normality: bool,
    require_arch_free: bool,
) -> CandidateValidation:
    if not candidate.success:
        return CandidateValidation(
            order=candidate.order,
            estimation_success=False,
            serially_adequate=False,
            adequate=False,
            residual_nobs=0,
            validation_lags=lags,
            alpha=alpha,
            residual_correlogram=None,
            mean_test=None,
            normality_test=None,
            ks_test=None,
            arch_test=None,
            required_lag_pvalues=tuple(),
            failed_checks=("estimation",),
            rationale="Candidate is ineligible because estimation failed.",
            error=candidate.error or "estimation failed",
        )

    try:
        corr = residual_correlogram(candidate.residuals, lags=lags, model_df=candidate.order[0] + candidate.order[2], alpha=alpha)
        finite_required = [float(p) for p, df in zip(corr.Prob, corr.DF) if int(df) > 0]
        finite_required_values = [p for p in finite_required if np.isfinite(p)]
        missing_required = any(not np.isfinite(p) for p in finite_required)
        serially_adequate = bool(finite_required_values) and all(p >= alpha for p in finite_required_values) and not missing_required
        failed: list[str] = []
        if not serially_adequate:
            failed.append("serial_correlation")

        mean_result = mean_zero_test(candidate.residuals, alpha=alpha)
        normality_result = jarque_bera(candidate.residuals, alpha=alpha) if require_normality else None
        ks_result = normality_ks(candidate.residuals, alpha=alpha) if require_normality else None
        arch_result = arch_test(candidate.residuals, lags=min(lags, max(1, candidate.residuals.size // 5)), alpha=alpha) if require_arch_free else None

        if require_mean_zero and mean_result.reject:
            failed.append("mean_zero")
        if require_normality and normality_result is not None and normality_result.reject:
            failed.append("normality")
        if require_normality and ks_result is not None and ks_result.reject:
            failed.append("normality_ks")
        if require_arch_free and arch_result is not None and arch_result.reject:
            failed.append("arch")

        adequate = not failed
        rationale = (
            "Candidate passes all configured residual-adequacy checks and is eligible for model selection."
            if adequate
            else "Candidate is rejected before model selection because: " + ", ".join(failed) + "."
        )
        return CandidateValidation(
            order=candidate.order,
            estimation_success=True,
            serially_adequate=serially_adequate,
            adequate=adequate,
            residual_nobs=corr.nobs,
            validation_lags=corr.nlags,
            alpha=alpha,
            residual_correlogram=corr,
            mean_test=mean_result,
            normality_test=normality_result,
            ks_test=ks_result,
            arch_test=arch_result,
            required_lag_pvalues=tuple(float(p) for p in finite_required),
            failed_checks=tuple(failed),
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001
        return CandidateValidation(
            order=candidate.order,
            estimation_success=True,
            serially_adequate=False,
            adequate=False,
            residual_nobs=0,
            validation_lags=lags,
            alpha=alpha,
            residual_correlogram=None,
            mean_test=None,
            normality_test=None,
            ks_test=None,
            arch_test=None,
            required_lag_pvalues=tuple(),
            failed_checks=("validation",),
            rationale="Candidate is ineligible because residual validation could not be completed.",
            error=str(exc),
        )


def validate_box_jenkins_candidates(
    estimation: BoxJenkinsEstimationResult,
    *,
    lags: int = 12,
    alpha: float = 0.05,
    require_mean_zero: bool = False,
    require_normality: bool = False,
    require_arch_free: bool = False,
) -> BoxJenkinsValidationResult:
    """Validate every estimated candidate and filter the eligible set.

    The mandatory adequacy rule is serial adequacy from the frozen Stage 8
    residual correlogram: every requested Ljung-Box lag with positive adjusted
    DF must have a finite p-value >= alpha. Other diagnostics are opt-in and
    can only make eligibility stricter.
    """
    if not isinstance(lags, int) or isinstance(lags, bool) or lags < 1:
        raise ValueError("lags must be a positive integer")
    if not 0 < float(alpha) < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    return BoxJenkinsValidationResult(
        tuple(
            _validate_candidate(
                candidate,
                lags=lags,
                alpha=alpha,
                require_mean_zero=require_mean_zero,
                require_normality=require_normality,
                require_arch_free=require_arch_free,
            )
            for candidate in estimation.candidates
        )
    )
