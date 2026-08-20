"""Box-Jenkins identification helpers.

Stage 9 identification is deliberately separate from final model selection:
it applies an explicit differencing rule, computes the frozen Stage 8 ACF/PACF
primitives, derives deterministic order hints, and generates a reproducible
candidate set. The existing :func:`identify` API remains backward compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd

from .correlation import ACFResult, PACFResult, acf, pacf
from .models import compare_orders
from .stationarity import dickey_fuller_sequential


@dataclass(frozen=True)
class BoxJenkinsIdentificationResult:
    """Auditable Stage 9.2 identification output."""

    original: np.ndarray
    transformed: np.ndarray
    differencing_order: int
    differencing_rule: str
    stationarity_decisions: tuple[str, ...]
    acf_result: ACFResult
    pacf_result: PACFResult
    acf_significant_lags: tuple[int, ...]
    pacf_significant_lags: tuple[int, ...]
    ar_order_hint: int
    ma_order_hint: int
    candidate_orders: tuple[tuple[int, int, int], ...]
    interpretation: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "original", np.asarray(self.original, dtype=float).copy())
        object.__setattr__(self, "transformed", np.asarray(self.transformed, dtype=float).copy())
        self.original.setflags(write=False)
        self.transformed.setflags(write=False)

    @property
    def ACF(self) -> ACFResult:
        return self.acf_result

    @property
    def PACF(self) -> PACFResult:
        return self.pacf_result

    @property
    def d(self) -> int:
        return self.differencing_order

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"p": p, "d": d, "q": q} for p, d, q in self.candidate_orders],
            columns=["p", "d", "q"],
        )


def _as_array(y) -> np.ndarray:
    if hasattr(y, "values"):
        values = np.asarray(y.values, dtype=float)
    else:
        values = np.asarray(list(y), dtype=float)
    if values.ndim != 1:
        raise ValueError("identification requires a univariate series")
    if values.size < 2:
        raise ValueError("identification requires at least two observations")
    if np.isnan(values).any():
        values = values[~np.isnan(values)]
    if not np.isfinite(values).all():
        raise ValueError("series must not contain infinite observations")
    if values.size < 2:
        raise ValueError("identification requires at least two non-missing observations")
    return values.astype(float, copy=True)


def _difference(values: np.ndarray, order: int) -> np.ndarray:
    result = values.copy()
    for _ in range(order):
        result = np.diff(result)
    if result.size < 2:
        raise ValueError("differencing leaves fewer than two usable observations")
    return result


def _choose_differencing(values: np.ndarray, *, d: int | None, max_d: int, alpha: float, whitening_lags: int):
    if d is not None:
        if not isinstance(d, int) or isinstance(d, bool) or d < 0 or d > max_d:
            raise ValueError("d must be an integer in [0, max_d]")
        return d, _difference(values, d), (f"d={d}: explicitly supplied by caller",), "explicit differencing"
    decisions = []
    transformed = values.copy()
    for current_d in range(max_d + 1):
        result = dickey_fuller_sequential(transformed, max_lags=None, autolag=None, alpha=alpha, whitening_lags=whitening_lags)
        decisions.append(f"d={current_d}: {result.selected.decision}; nature={result.nature}; common_lag={result.common_lag}")
        if result.selected.rejects_null or current_d == max_d:
            return current_d, transformed, tuple(decisions), "smallest d<=max_d for which the course-faithful sequential DF test rejects the unit-root null"
        transformed = _difference(transformed, 1)
    raise RuntimeError("automatic differencing failed to produce a result")


def _initial_contiguous_significant(result, *, max_order: int) -> tuple[int, tuple[int, ...]]:
    significant = np.asarray(result.significant(), dtype=bool)
    lags = np.asarray(result.lags, dtype=int)
    significant_lags = tuple(int(lag) for lag in lags[1:] if significant[lag])
    available_lags = int(lags.max()) if lags.size else 0
    hint = 0
    for lag in range(1, min(max_order, available_lags) + 1):
        if significant[lag]:
            hint = lag
        else:
            break
    return hint, significant_lags


def _candidate_orders(d: int, *, p_hint: int, q_hint: int, max_p: int, max_q: int):
    p_cap = min(max_p, max(1, p_hint)) if max_p else 0
    q_cap = min(max_q, max(1, q_hint)) if max_q else 0
    candidates = {(0, d, 0), *((p, d, 0) for p in range(1, p_cap + 1)), *((0, d, q) for q in range(1, q_cap + 1)), *((p, d, q) for p, q in product(range(1, p_cap + 1), range(1, q_cap + 1)))}
    return tuple(sorted(candidates))


def identify_box_jenkins(y, *, d: int | None = None, max_d: int = 2, nlags: int = 24, max_p: int = 2, max_q: int = 2, alpha: float = 0.05, whitening_lags: int = 12) -> BoxJenkinsIdentificationResult:
    if not isinstance(max_d, int) or isinstance(max_d, bool) or not 0 <= max_d <= 2:
        raise ValueError("max_d must be an integer in [0, 2]")
    if not isinstance(nlags, int) or isinstance(nlags, bool) or nlags < 1:
        raise ValueError("nlags must be a positive integer")
    if not isinstance(max_p, int) or isinstance(max_p, bool) or max_p < 0:
        raise ValueError("max_p must be a non-negative integer")
    if not isinstance(max_q, int) or isinstance(max_q, bool) or max_q < 0:
        raise ValueError("max_q must be a non-negative integer")
    if not 0 < float(alpha) < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if not isinstance(whitening_lags, int) or isinstance(whitening_lags, bool) or whitening_lags < 1:
        raise ValueError("whitening_lags must be a positive integer")
    original = _as_array(y)
    d_order, transformed, decisions, rule = _choose_differencing(original, d=d, max_d=max_d, alpha=alpha, whitening_lags=whitening_lags)
    ac = acf(transformed, nlags=nlags, alpha=alpha)
    pc = pacf(transformed, nlags=nlags, alpha=alpha)
    ac_cutoff, ac_sig = _initial_contiguous_significant(ac, max_order=max_q)
    pc_cutoff, pc_sig = _initial_contiguous_significant(pc, max_order=max_p)
    candidates = _candidate_orders(d_order, p_hint=pc_cutoff, q_hint=ac_cutoff, max_p=max_p, max_q=max_q)
    interpretation = f"After d={d_order} differencing, PACF significant lags={list(pc_sig)} and ACF significant lags={list(ac_sig)}. Initial contiguous PACF cutoff={pc_cutoff} suggests AR({pc_cutoff}) and ACF cutoff={ac_cutoff} suggests MA({ac_cutoff}). Generated {len(candidates)} deterministic candidate orders; no model is selected at this stage."
    return BoxJenkinsIdentificationResult(original=original, transformed=transformed, differencing_order=d_order, differencing_rule=rule, stationarity_decisions=decisions, acf_result=ac, pacf_result=pc, acf_significant_lags=ac_sig, pacf_significant_lags=pc_sig, ar_order_hint=pc_cutoff, ma_order_hint=ac_cutoff, candidate_orders=candidates, interpretation=interpretation)


def identify(y, *, nlags: int = 24, alpha: float = 0.05):
    ac = acf(y, nlags=nlags, alpha=alpha)
    pc = pacf(y, nlags=nlags, alpha=alpha)
    ac_sig = [int(k) for k in ac.lags[1:] if ac.significant()[k]]
    pc_sig = [int(k) for k in pc.lags[1:] if pc.significant()[k]]
    return {"ACF": ac, "PACF": pc, "ACF significant lags": ac_sig, "PACF significant lags": pc_sig, "AR order clue": max(pc_sig) if pc_sig else 0, "MA order clue": max(ac_sig) if ac_sig else 0, "interpretation": _interpret(ac_sig, pc_sig)}


def grid_search(y, *, p_max: int = 4, d_values: Iterable[int] = (0, 1), q_max: int = 4) -> pd.DataFrame:
    orders = list(product(range(p_max + 1), d_values, range(q_max + 1)))
    orders = [o for o in orders if o != (0, 0, 0)]
    return compare_orders(y, orders)


def _interpret(ac_sig: list[int], pac_sig: list[int]) -> str:
    if not ac_sig and not pac_sig:
        return "ACF and PACF show no significant non-zero lags; a white-noise or very low-order process should be considered."
    if pac_sig and not ac_sig:
        return f"PACF has significant lags {pac_sig}; this pattern is compatible with an AR component, subject to estimation and validation."
    if ac_sig and not pac_sig:
        return f"ACF has significant lags {ac_sig}; this pattern is compatible with an MA component, subject to estimation and validation."
    return f"Both ACF and PACF contain significant lags (ACF={ac_sig}, PACF={pac_sig}); consider mixed ARMA candidates and compare information criteria."
