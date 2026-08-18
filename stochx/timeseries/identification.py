"""Box-Jenkins identification helpers."""

from __future__ import annotations

from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd

from .correlation import acf, pacf
from .models import compare_orders


def identify(y, *, nlags: int = 24, alpha: float = 0.05) -> dict[str, object]:
    """Return ACF/PACF tables and simple AR/MA order clues."""
    ac = acf(y, nlags=nlags, alpha=alpha)
    pc = pacf(y, nlags=nlags, alpha=alpha)
    ac_sig = [int(k) for k in ac.lags[1:] if ac.significant()[k]]
    pc_sig = [int(k) for k in pc.lags[1:] if pc.significant()[k]]
    return {
        "ACF": ac,
        "PACF": pc,
        "ACF significant lags": ac_sig,
        "PACF significant lags": pc_sig,
        "AR order clue": max(pc_sig) if pc_sig else 0,
        "MA order clue": max(ac_sig) if ac_sig else 0,
        "interpretation": _interpret(ac_sig, pc_sig),
    }


def grid_search(y, *, p_max: int = 4, d_values: Iterable[int] = (0, 1), q_max: int = 4) -> pd.DataFrame:
    """Compare a Box-Jenkins grid using AIC, BIC and HQ."""
    orders = list(product(range(p_max + 1), d_values, range(q_max + 1)))
    orders = [o for o in orders if o != (0, 0, 0)]
    return compare_orders(y, orders)


def _interpret(ac_sig: list[int], pac_sig: list[int]) -> str:
    """Create a compact course-style ACF/PACF interpretation."""
    if not ac_sig and not pac_sig:
        return "ACF and PACF show no significant non-zero lags; a white-noise or very low-order process should be considered."
    if pac_sig and not ac_sig:
        return f"PACF has significant lags {pac_sig}; this pattern is compatible with an AR component, subject to estimation and validation."
    if ac_sig and not pac_sig:
        return f"ACF has significant lags {ac_sig}; this pattern is compatible with an MA component, subject to estimation and validation."
    return f"Both ACF and PACF contain significant lags (ACF={ac_sig}, PACF={pac_sig}); consider mixed ARMA candidates and compare information criteria."
