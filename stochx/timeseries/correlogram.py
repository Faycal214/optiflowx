"""EViews-style correlogram and Ljung-Box diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .correlation import acf, pacf


@dataclass(frozen=True)
class LjungBoxResult:
    """Cumulative Ljung-Box statistics and chi-square probabilities."""

    lags: np.ndarray
    q_stat: np.ndarray
    pvalues: np.ndarray
    df: np.ndarray
    nobs: int
    model_df: int

    def table(self) -> pd.DataFrame:
        """Return an EViews-style Q-statistic table."""
        return pd.DataFrame(
            {
                "Lag": self.lags,
                "Q-Stat": self.q_stat,
                "Prob.": self.pvalues,
                "DF": self.df,
            }
        )

    def summary(self) -> str:
        """Return a compact EViews-style Q-statistic summary."""
        rows = self.table()
        lines = [
            f"Ljung-Box Q-statistics (nobs={self.nobs}, model_df={self.model_df})",
            "Lag       Q-Stat       Prob.       DF",
            "------------------------------------------",
        ]
        for row in rows.itertuples(index=False):
            prob = "NA" if np.isnan(row[2]) else f"{row[2]:.4f}"
            lines.append(f"{int(row[0]):>3d}   {row[1]:>11.4f}   {prob:>8}   {int(row[3]):>3d}")
        return "\n".join(lines)

    def interpret(self, alpha: float = 0.05) -> str:
        """Interpret the final valid Ljung-Box probability."""
        valid = np.flatnonzero(np.isfinite(self.pvalues))
        if valid.size == 0:
            return "No Ljung-Box probability is defined because the adjusted degrees of freedom are non-positive."
        idx = int(valid[-1])
        lag = int(self.lags[idx])
        pvalue = float(self.pvalues[idx])
        if pvalue < alpha:
            return (
                f"At lag {lag}, the Ljung-Box test rejects the no-autocorrelation null "
                f"at the {alpha:.0%} level (p={pvalue:.4g})."
            )
        return (
            f"At lag {lag}, the Ljung-Box test does not reject the no-autocorrelation "
            f"null at the {alpha:.0%} level (p={pvalue:.4g})."
        )


def ljung_box(
    ac_values: np.ndarray,
    *,
    nobs: int,
    model_df: int = 0,
    nlags: int | None = None,
) -> LjungBoxResult:
    """Compute cumulative Ljung-Box Q statistics from autocorrelations.

    For each lag k, the statistic is

        Q_k = n(n+2) * sum_{j=1}^k rho_j^2 / (n-j)

    with chi-square degrees of freedom ``k - model_df``. Probabilities are
    undefined (NaN) whenever the adjusted degrees of freedom are non-positive.
    """
    values = np.asarray(ac_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("ac_values must be one-dimensional")
    if nobs < 2:
        raise ValueError("nobs must be at least 2")
    if not isinstance(model_df, int) or model_df < 0:
        raise ValueError("model_df must be a non-negative integer")
    if nlags is None:
        nlags = values.size - 1
    if not isinstance(nlags, int) or nlags < 1:
        raise ValueError("nlags must be a positive integer")
    if nlags >= nobs:
        raise ValueError("nlags must be smaller than nobs")
    if values.size < nlags + 1:
        raise ValueError("ac_values must contain lag 0 through nlags")

    lags = np.arange(1, nlags + 1, dtype=int)
    q_stats = np.empty(nlags, dtype=float)
    df = lags.astype(int) - model_df
    pvalues = np.full(nlags, np.nan, dtype=float)
    cumulative = 0.0
    for i, lag in enumerate(lags):
        cumulative += values[lag] ** 2 / (nobs - lag)
        q_stats[i] = nobs * (nobs + 2) * cumulative
        if df[i] > 0:
            pvalues[i] = float(chi2.sf(q_stats[i], int(df[i])))

    return LjungBoxResult(lags, q_stats, pvalues, df, nobs, model_df)


def correlogram(series, *, nlags: int = 36, model_df: int = 0, alpha: float = 0.05) -> pd.DataFrame:
    """Return AC, PAC, Ljung-Box Q-Stat and Prob columns matching Stage 8.1."""
    if not isinstance(model_df, int) or model_df < 0:
        raise ValueError("model_df must be a non-negative integer")
    ac = acf(series, nlags=nlags, alpha=alpha)
    pc = pacf(series, nlags=nlags, alpha=alpha)
    lb = ljung_box(ac.values, nobs=ac.nobs, model_df=model_df, nlags=nlags)
    ac_sig = ac.significant()
    pc_sig = pc.significant()
    rows = []
    for i, lag in enumerate(lb.lags):
        rows.append(
            {
                "Lag": int(lag),
                "AC": float(ac.values[lag]),
                "PAC": float(pc.values[lag]),
                "Q-Stat": float(lb.q_stat[i]),
                "Prob.": float(lb.pvalues[i]) if np.isfinite(lb.pvalues[i]) else np.nan,
                "DF": int(lb.df[i]),
                "AC Significant": bool(ac_sig[lag]),
                "PAC Significant": bool(pc_sig[lag]),
            }
        )
    return pd.DataFrame(rows)
