"""Stage 9.5 deterministic Box-Jenkins model selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .box_jenkins_estimation import EstimatedCandidate
from .box_jenkins_validation import BoxJenkinsValidationResult, CandidateValidation

SelectionCriterion = Literal["aic", "sc", "bic", "hq"]


@dataclass(frozen=True)
class BoxJenkinsSelectionResult:
    """Auditable deterministic selection outcome for validated candidates."""

    criterion: str
    tie_tolerance: float
    eligible_orders: tuple[tuple[int, int, int], ...]
    ranked_candidates: tuple[EstimatedCandidate, ...]
    selected: EstimatedCandidate | None
    status: str
    rationale: str

    @property
    def selected_order(self) -> tuple[int, int, int] | None:
        return None if self.selected is None else self.selected.order

    @property
    def has_selection(self) -> bool:
        return self.selected is not None

    def table(self) -> pd.DataFrame:
        """Return the deterministic ranking table used for the decision."""
        rows = []
        for rank, candidate in enumerate(self.ranked_candidates, start=1):
            value = _criterion_value(candidate, self.criterion)
            rows.append(
                {
                    "Rank": rank,
                    "p": candidate.order[0],
                    "d": candidate.order[1],
                    "q": candidate.order[2],
                    "Parameters": candidate.params.size,
                    "Criterion": value,
                    "AIC": candidate.aic,
                    "SC": candidate.bic,
                    "HQ": candidate.hq,
                    "Selected": self.selected is not None and candidate.order == self.selected.order,
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "Rank", "p", "d", "q", "Parameters", "Criterion",
                "AIC", "SC", "HQ", "Selected",
            ],
        )


def _normalize_criterion(criterion: str) -> str:
    value = str(criterion).strip().lower()
    aliases = {"aic": "aic", "sc": "sc", "bic": "sc", "hq": "hq"}
    if value not in aliases:
        raise ValueError("criterion must be one of 'aic', 'sc'/'bic', or 'hq'")
    return aliases[value]


def _criterion_value(candidate: EstimatedCandidate, criterion: str) -> float:
    value = {"aic": candidate.aic, "sc": candidate.bic, "hq": candidate.hq}[criterion]
    return float(value)


def _rank_candidates(candidates: tuple[EstimatedCandidate, ...], criterion: str, tie_tolerance: float) -> tuple[EstimatedCandidate, ...]:
    ordered = sorted(candidates, key=lambda c: (_criterion_value(c, criterion), int(c.params.size), c.order))
    if len(ordered) < 2 or tie_tolerance == 0:
        return tuple(ordered)

    ranked: list[EstimatedCandidate] = []
    remaining = list(ordered)
    while remaining:
        reference = _criterion_value(remaining[0], criterion)
        tied = [c for c in remaining if abs(_criterion_value(c, criterion) - reference) <= tie_tolerance]
        tied.sort(key=lambda c: (int(c.params.size), c.order))
        ranked.extend(tied)
        tied_ids = {id(c) for c in tied}
        remaining = [c for c in remaining if id(c) not in tied_ids]
    return tuple(ranked)


def _coerce_estimated(candidate: CandidateValidation) -> EstimatedCandidate | None:
    """Return the immutable estimation snapshot attached by Stage 9.4."""
    return candidate.estimated_candidate


def select_box_jenkins_model(
    validation: BoxJenkinsValidationResult,
    *,
    criterion: SelectionCriterion = "aic",
    tie_tolerance: float = 1e-8,
) -> BoxJenkinsSelectionResult:
    """Select one model from the Stage 9.4 eligible candidates.

    Selection never re-estimates or re-validates candidates. Adequacy is a
    hard eligibility gate; information criteria are applied only afterward.
    """
    normalized = _normalize_criterion(criterion)
    if not np.isfinite(float(tie_tolerance)) or float(tie_tolerance) < 0:
        raise ValueError("tie_tolerance must be a finite non-negative number")

    eligible_validations = tuple(c for c in validation.candidates if c.eligible)
    eligible = tuple(c.model for c in eligible_validations if c.model is not None)
    eligible_orders = tuple(c.order for c in eligible_validations)
    if len(eligible) != len(eligible_validations):
        return BoxJenkinsSelectionResult(
            criterion=normalized,
            tie_tolerance=float(tie_tolerance),
            eligible_orders=eligible_orders,
            ranked_candidates=tuple(),
            selected=None,
            status="missing_estimation_snapshot",
            rationale="One or more eligible validation records has no underlying estimation snapshot; selection is not performed.",
        )
    if not eligible:
        return BoxJenkinsSelectionResult(
            criterion=normalized,
            tie_tolerance=float(tie_tolerance),
            eligible_orders=eligible_orders,
            ranked_candidates=tuple(),
            selected=None,
            status="no_adequate_model",
            rationale="No candidate passed Stage 9.4 residual validation; selection is not performed.",
        )

    ranked = _rank_candidates(eligible, normalized, float(tie_tolerance))
    selected = ranked[0]
    selected_value = _criterion_value(selected, normalized)
    peers = [c for c in ranked if abs(_criterion_value(c, normalized) - selected_value) <= float(tie_tolerance)]
    rationale = (
        f"Selected {selected.order} using {normalized.upper()}={selected_value:.12g}. "
        f"Only Stage 9.4 eligible candidates were considered; {len(eligible)} eligible candidate(s) were ranked. "
    )
    if len(peers) > 1:
        rationale += (
            f"The winning criterion value was tied within tolerance {tie_tolerance:g}; "
            f"parameter-count parsimony selected {selected.params.size} estimated parameter(s), "
            f"then lexicographic order {selected.order} as the final deterministic tie-break."
        )
    else:
        rationale += "No information-criterion tie within the configured tolerance required a parsimony tie-break."

    return BoxJenkinsSelectionResult(
        criterion=normalized,
        tie_tolerance=float(tie_tolerance),
        eligible_orders=eligible_orders,
        ranked_candidates=ranked,
        selected=selected,
        status="selected",
        rationale=rationale,
    )
