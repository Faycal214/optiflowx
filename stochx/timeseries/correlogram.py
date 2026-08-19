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


@dataclass(frozen=True)
class CorrelogramResult:
    """Unified, auditable EViews-style correlogram result.

    The object contains all numeric series needed to reproduce the displayed
    correlogram: AC, PAC, Ljung-Box Q-statistics, adjusted degrees of freedom,
    probabilities, and the shared run metadata.
    """

    lags: np.ndarray
    ac: np.ndarray
    pac: np.ndarray
    q_stat: np.ndarray
    pvalues: np.ndarray
    df: np.ndarray
    nobs: int
    nlags: int
    model_df: int
    alpha: float
    series_name: str = "series"
    ac_lower: np.ndarray | None = None
    ac_upper: np.ndarray | None = None
    pac_lower: np.ndarray | None = None
    pac_upper: np.ndarray | None = None

    @property
    def DF(self) -> np.ndarray:
        """Adjusted Ljung-Box degrees of freedom, EViews-style table field."""
        return self.df

    @property
    def Q_Stat(self) -> np.ndarray:
        """Ljung-Box Q-statistics, EViews-style table field."""
        return self.q_stat

    @property
    def Prob(self) -> np.ndarray:
        """Ljung-Box chi-square probabilities, EViews-style table field."""
        return self.pvalues

    def table(self) -> pd.DataFrame:
        """Return the unified EViews-style correlogram table."""
        rows = {
            "Lag": self.lags,
            "AC": self.ac,
            "PAC": self.pac,
            "Q-Stat": self.q_stat,
            "Prob.": self.pvalues,
            "DF": self.df,
        }
        if self.ac_lower is not None:
            rows["AC Lower"] = self.ac_lower
        if self.ac_upper is not None:
            rows["AC Upper"] = self.ac_upper
        if self.pac_lower is not None:
            rows["PAC Lower"] = self.pac_lower
        if self.pac_upper is not None:
            rows["PAC Upper"] = self.pac_upper
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Return a compact EViews-style correlogram summary."""
        lines = [
            f"Correlogram for {self.series_name}",
            f"Included observations: {self.nobs}",
            f"Lags: 1 to {self.nlags}; model_df={self.model_df}; alpha={self.alpha:g}",
            "Lag       AC        PAC       Q-Stat       Prob.     DF",
            "------------------------------------------------------------",
        ]
        for row in self.table().itertuples(index=False):
            prob = "NA" if pd.isna(row[4]) else f"{row[4]:.4f}"
            lines.append(
                f"{int(row[0]):>3d}   {row[1]:>9.4f}  {row[2]:>9.4f}  "
                f"{row[3]:>11.4f}  {prob:>8}  {int(row[5]):>3d}"
            )
        return "\n".join(lines)

    def interpret(self) -> str:
        """Interpret the final valid Ljung-Box probability."""
        valid = np.flatnonzero(np.isfinite(self.pvalues))
        if valid.size == 0:
            return "No Ljung-Box probability is defined because the adjusted degrees of freedom are non-positive."
        idx = int(valid[-1])
        lag = int(self.lags[idx])
        pvalue = float(self.pvalues[idx])
        decision = "rejects" if pvalue < self.alpha else "does not reject"
        return (
            f"At lag {lag}, the Ljung-Box test {decision} the no-autocorrelation null "
            f"at the {self.alpha:.0%} level (p={pvalue:.4g})."
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


def correlogram(
    series,
    *,
    nlags: int = 36,
    model_df: int = 0,
    alpha: float = 0.05,
) -> CorrelogramResult:
    """Return a unified EViews-style correlogram result."""
    if not isinstance(model_df, int) or model_df < 0:
        raise ValueError("model_df must be a non-negative integer")
    ac = acf(series, nlags=nlags, alpha=alpha)
    pc = pacf(series, nlags=nlags, alpha=alpha)
    lb = ljung_box(ac.values, nobs=ac.nobs, model_df=model_df, nlags=nlags)
    ac_sig = ac.significant()
    pc_sig = pc.significant()
    lags = lb.lags
    return CorrelogramResult(
        lags=lags,
        ac=ac.values[lags],
        pac=pc.values[lags],
        q_stat=lb.q_stat,
        pvalues=lb.pvalues,
        df=lb.df,
        nobs=ac.nobs,
        nlags=nlags,
        model_df=model_df,
        alpha=alpha,
        series_name=getattr(ac, "series_name", "series"),
        ac_lower=ac.lower[lags],
        ac_upper=ac.upper[lags],
        pac_lower=pc.lower[lags],
        pac_upper=pc.upper[lags],
    )
