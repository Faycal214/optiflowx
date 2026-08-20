"""Stage 9.3 Box-Jenkins candidate estimation.

This module composes the existing StochX AR/MA/ARMA/ARIMA estimators.  It
records auditable numerical outputs for every deterministic candidate without
changing the underlying model-fitting implementations or the frozen Stage 8
correlogram contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .models import TSResult, estimate


@dataclass(frozen=True)
class EstimatedCandidate:
    """Immutable numerical snapshot for one candidate model."""

    order: tuple[int, int, int]
    success: bool
    model_name: str
    estimation_nobs: int | None
    params: np.ndarray
    standard_errors: np.ndarray
    tvalues: np.ndarray
    pvalues: np.ndarray
    log_likelihood: float
    sigma_sq: float
    aic: float
    bic: float
    hq: float
    ar_roots: np.ndarray
    ma_roots: np.ndarray
    converged: bool
    residuals: np.ndarray
    ts_result: TSResult | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        order = tuple(int(v) for v in self.order)
        if len(order) != 3 or min(order) < 0:
            raise ValueError("order must be a non-negative (p, d, q) triple")
        object.__setattr__(self, "order", order)
        for name in (
            "params",
            "standard_errors",
            "tvalues",
            "pvalues",
            "ar_roots",
            "ma_roots",
            "residuals",
        ):
            array = np.asarray(getattr(self, name))
            array = array.copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if self.estimation_nobs is not None:
            object.__setattr__(self, "estimation_nobs", int(self.estimation_nobs))
        if self.error is not None:
            object.__setattr__(self, "error", str(self.error))

    @property
    def coefficient_names(self) -> tuple[str, ...]:
        if self.ts_result is None:
            return tuple()
        names = getattr(self.ts_result.result, "param_names", None)
        if names is None:
            names = getattr(self.ts_result.result, "params", pd.Series(dtype=float)).index if hasattr(getattr(self.ts_result.result, "params", None), "index") else []
        return tuple(str(name) for name in names)


@dataclass(frozen=True)
class BoxJenkinsEstimationResult:
    """Auditable collection of successfully/unsuccessfully estimated candidates."""

    candidates: tuple[EstimatedCandidate, ...]

    @property
    def successful(self) -> tuple[EstimatedCandidate, ...]:
        return tuple(candidate for candidate in self.candidates if candidate.success)

    @property
    def failed(self) -> tuple[EstimatedCandidate, ...]:
        return tuple(candidate for candidate in self.candidates if not candidate.success)

    @property
    def orders(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(candidate.order for candidate in self.candidates)

    def table(self) -> pd.DataFrame:
        """Return a deterministic candidate-estimation audit table."""
        rows = []
        for candidate in self.candidates:
            rows.append(
                {
                    "p": candidate.order[0],
                    "d": candidate.order[1],
                    "q": candidate.order[2],
                    "success": candidate.success,
                    "model": candidate.model_name,
                    "nobs": candidate.estimation_nobs,
                    "LogLik": candidate.log_likelihood,
                    "SIGMASQ": candidate.sigma_sq,
                    "AIC": candidate.aic,
                    "SC": candidate.bic,
                    "HQ": candidate.hq,
                    "converged": candidate.converged,
                    "error": candidate.error,
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "p", "d", "q", "success", "model", "nobs", "LogLik",
                "SIGMASQ", "AIC", "SC", "HQ", "converged", "error",
            ],
        )


def _float_attr(result, name: str, default: float = np.nan) -> float:
    value = getattr(result, name, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value


def _array_attr(result, name: str) -> np.ndarray:
    value = getattr(result, name, np.asarray([], dtype=float))
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    return np.asarray(value, dtype=float).copy()


def _converged(result) -> bool:
    converged = getattr(result, "mle_retvals", None)
    if isinstance(converged, dict) and "converged" in converged:
        return bool(converged["converged"])
    flag = getattr(result, "converged", None)
    if flag is not None:
        return bool(flag)
    return True


def _sigma_sq(result, residuals: np.ndarray) -> float:
    for name in ("sigma2", "scale"):
        value = getattr(result, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    valid = residuals[np.isfinite(residuals)]
    return float(np.mean(valid**2)) if valid.size else float("nan")


def _residuals(result) -> np.ndarray:
    value = getattr(result, "resid", np.asarray([], dtype=float))
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    return np.asarray(value, dtype=float).copy()


def _fit_candidate(y, order: tuple[int, int, int]) -> EstimatedCandidate:
    p, d, q = order
    try:
        fitted = estimate(y, p=p, d=d, q=q)
        result = fitted.result
        residuals = _residuals(result)
        params = _array_attr(result, "params")
        standard_errors = _array_attr(result, "bse")
        tvalues = _array_attr(result, "tvalues")
        pvalues = _array_attr(result, "pvalues")
        roots = fitted.roots()
        stats = fitted.statistics()
        return EstimatedCandidate(
            order=order,
            success=True,
            model_name=fitted.model_name,
            estimation_nobs=int(getattr(result, "nobs", np.isfinite(residuals).sum())),
            params=params,
            standard_errors=standard_errors,
            tvalues=tvalues,
            pvalues=pvalues,
            log_likelihood=_float_attr(result, "llf", stats.get("Log likelihood", np.nan)),
            sigma_sq=_sigma_sq(result, residuals),
            aic=_float_attr(result, "aic", stats.get("Akaike info criterion", np.nan)),
            bic=_float_attr(result, "bic", stats.get("Schwarz criterion", np.nan)),
            hq=_float_attr(result, "hqic", stats.get("Hannan-Quinn criterion", np.nan)),
            ar_roots=np.asarray(roots["AR roots"], dtype=complex),
            ma_roots=np.asarray(roots["MA roots"], dtype=complex),
            converged=_converged(result),
            residuals=residuals,
            ts_result=fitted,
        )
    except Exception as exc:  # noqa: BLE001
        return EstimatedCandidate(
            order=order,
            success=False,
            model_name="",
            estimation_nobs=None,
            params=np.asarray([], dtype=float),
            standard_errors=np.asarray([], dtype=float),
            tvalues=np.asarray([], dtype=float),
            pvalues=np.asarray([], dtype=float),
            log_likelihood=np.nan,
            sigma_sq=np.nan,
            aic=np.nan,
            bic=np.nan,
            hq=np.nan,
            ar_roots=np.asarray([], dtype=complex),
            ma_roots=np.asarray([], dtype=complex),
            converged=False,
            residuals=np.asarray([], dtype=float),
            ts_result=None,
            error=str(exc),
        )


def estimate_box_jenkins_candidates(
    y,
    candidate_orders: Iterable[tuple[int, int, int]],
) -> BoxJenkinsEstimationResult:
    """Estimate candidates in deterministic order using the existing estimators.

    Each candidate is attempted independently.  One failed candidate is stored
    as a structured failure and does not prevent the remaining candidates from
    being estimated.
    """
    orders = tuple((int(p), int(d), int(q)) for p, d, q in candidate_orders)
    if len(set(orders)) != len(orders):
        raise ValueError("candidate_orders must not contain duplicates")
    if any(len(order) != 3 or min(order) < 0 for order in orders):
        raise ValueError("candidate_orders must contain non-negative (p, d, q) triples")
    return BoxJenkinsEstimationResult(tuple(_fit_candidate(y, order) for order in orders))
