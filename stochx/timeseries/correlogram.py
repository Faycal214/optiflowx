"""EViews-style combined autocorrelation/partial-correlation output."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .correlation import acf, pacf


def correlogram(series, *, nlags: int = 36, model_df: int = 0, alpha: float = 0.05) -> pd.DataFrame:
    """Return AC, PAC, Q-Stat and Prob columns matching the course's EViews output."""
    ac = acf(series, nlags=nlags, alpha=alpha)
    pc = pacf(series, nlags=nlags, alpha=alpha)
    n = ac.nobs
    rows = []
    q = 0.0
    for k in range(1, nlags + 1):
        r = float(ac.values[k])
        q += r * r / max(n - k, 1)
        q_stat = n * (n + 2) * q
        df = max(k - model_df, 1)
        pvalue = float(chi2.sf(q_stat, df)) if k > model_df else np.nan
        rows.append({
            "Lag": k,
            "AC": r,
            "PAC": float(pc.values[k]),
            "Q-Stat": q_stat,
            "Prob.": pvalue,
            "AC Significant": bool(ac.significant()[k]),
            "PAC Significant": bool(pc.significant()[k]),
        })
    return pd.DataFrame(rows)
