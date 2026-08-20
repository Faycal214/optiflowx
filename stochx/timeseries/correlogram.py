"""EViews-style correlogram and Ljung-Box diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .correlation import EVIEWS_BAND_METHOD, EVIEWS_BAND_MULTIPLIER, acf, pacf


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
        return pd.DataFrame({"Lag": self.lags, "Q-Stat": self.q_stat, "Prob.": self.pvalues, "DF": self.df})

    def summary(self) -> str:
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
        valid = np.flatnonzero(np.isfinite(self.pvalues))
        if valid.size == 0:
            return "No Ljung-Box probability is defined because the adjusted degrees of freedom are non-positive."
        idx = int(valid[-1])
        lag = int(self.lags[idx])
        pvalue = float(self.pvalues[idx])
        if pvalue < alpha:
            return f"At lag {lag}, the Ljung-Box test rejects the no-autocorrelation null at the {alpha:.0%} level (p={pvalue:.4g})."
        return f"At lag {lag}, the Ljung-Box test does not reject the no-autocorrelation null at the {alpha:.0%} level (p={pvalue:.4g})."


@dataclass(frozen=True)
class CorrelogramResult:
    """Stable public result contract for an EViews-style correlogram.

    Canonical fields are snake_case. EViews-facing aliases are provided as
    properties (``AC``, ``PAC``, ``Q_Stat``, ``Prob``, ``DF``).
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
    missing_count: int = 0
    ac_lower: np.ndarray | None = None
    ac_upper: np.ndarray | None = None
    pac_lower: np.ndarray | None = None
    pac_upper: np.ndarray | None = None
    band_multiplier: float = EVIEWS_BAND_MULTIPLIER
    band_method: str = EVIEWS_BAND_METHOD

    def __post_init__(self) -> None:
        if not isinstance(self.nobs, (int, np.integer)) or isinstance(self.nobs, bool) or self.nobs < 2:
            raise ValueError("nobs must be an integer >= 2")
        if not isinstance(self.nlags, (int, np.integer)) or isinstance(self.nlags, bool) or not 1 <= self.nlags < self.nobs:
            raise ValueError("nlags must be an integer satisfying 1 <= nlags < nobs")
        if not isinstance(self.model_df, (int, np.integer)) or isinstance(self.model_df, bool) or self.model_df < 0:
            raise ValueError("model_df must be a non-negative integer")
        if not 0 < float(self.alpha) < 1:
            raise ValueError("alpha must lie strictly between 0 and 1")
        if not isinstance(self.series_name, str) or not self.series_name.strip():
            raise ValueError("series_name must be a non-empty string")
        if not isinstance(self.missing_count, (int, np.integer)) or isinstance(self.missing_count, bool) or self.missing_count < 0:
            raise ValueError("missing_count must be a non-negative integer")
        if self.missing_count + self.nobs < 1:
            raise ValueError("nobs and missing_count are inconsistent")
        if not np.isfinite(float(self.band_multiplier)) or float(self.band_multiplier) <= 0:
            raise ValueError("band_multiplier must be a positive finite number")
        if not isinstance(self.band_method, str) or not self.band_method.strip():
            raise ValueError("band_method must be a non-empty string")

        arrays = {
            "lags": self.lags,
            "ac": self.ac,
            "pac": self.pac,
            "q_stat": self.q_stat,
            "pvalues": self.pvalues,
            "df": self.df,
        }
        normalized: dict[str, np.ndarray] = {}
        for name, value in arrays.items():
            array = np.asarray(value)
            if array.ndim != 1 or array.size != self.nlags:
                raise ValueError(f"{name} must be one-dimensional with length nlags")
            normalized[name] = array.copy()

        expected_lags = np.arange(1, self.nlags + 1, dtype=int)
        if not np.array_equal(normalized["lags"], expected_lags):
            raise ValueError("lags must be exactly 1..nlags")
        if not np.isfinite(normalized["ac"]).all() or not np.isfinite(normalized["pac"]).all():
            raise ValueError("AC and PAC values must be finite")
        if not np.isfinite(normalized["q_stat"]).all() or not np.isfinite(normalized["df"]).all():
            raise ValueError("Q-Stat and DF values must be finite")
        if np.any(normalized["df"] != normalized["lags"] - self.model_df):
            raise ValueError("DF must equal Lag - model_df")
        probabilities = normalized["pvalues"]
        invalid_prob = ~np.isfinite(probabilities) & ~np.isnan(probabilities)
        if invalid_prob.any() or np.any((probabilities[np.isfinite(probabilities)] < 0) | (probabilities[np.isfinite(probabilities)] > 1)):
            raise ValueError("Prob. values must be in [0, 1] or NaN")

        bands = {
            "ac_lower": self.ac_lower,
            "ac_upper": self.ac_upper,
            "pac_lower": self.pac_lower,
            "pac_upper": self.pac_upper,
        }
        any_band = any(value is not None for value in bands.values())
        if any_band:
            if not all(value is not None for value in bands.values()):
                raise ValueError("all AC/PAC lower and upper band arrays must be supplied together")
            for name, value in bands.items():
                array = np.asarray(value)
                if array.ndim != 1 or array.size != self.nlags:
                    raise ValueError(f"{name} must be one-dimensional with length nlags")
                if not np.isfinite(array).all():
                    raise ValueError(f"{name} must contain only finite values")
                normalized[name] = array.copy()

        for name, array in normalized.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)

        object.__setattr__(self, "nobs", int(self.nobs))
        object.__setattr__(self, "nlags", int(self.nlags))
        object.__setattr__(self, "model_df", int(self.model_df))
        object.__setattr__(self, "missing_count", int(self.missing_count))
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "band_multiplier", float(self.band_multiplier))

    @property
    def AC(self) -> np.ndarray:
        return self.ac

    @property
    def PAC(self) -> np.ndarray:
        return self.pac

    @property
    def DF(self) -> np.ndarray:
        return self.df

    @property
    def Q_Stat(self) -> np.ndarray:
        return self.q_stat

    @property
    def QStat(self) -> np.ndarray:
        return self.q_stat

    @property
    def Prob(self) -> np.ndarray:
        return self.pvalues

    @property
    def PValues(self) -> np.ndarray:
        return self.pvalues

    @property
    def columns(self) -> pd.Index:
        """Backward-compatible DataFrame-style column index.

        Stage 8 returns a structured ``CorrelogramResult`` rather than a
        DataFrame. This read-only projection keeps legacy callers that inspect
        ``correlogram(...).columns`` working while directing data consumers to
        ``result.table()``.
        """
        return self.table().columns.copy()

    @property
    def band_standard_error(self) -> float:
        return 1.0 / np.sqrt(self.nobs)

    @property
    def band_half_width(self) -> float:
        return self.band_multiplier * self.band_standard_error

    @property
    def band_confidence_level(self) -> float:
        from scipy.stats import norm
        return float(2.0 * norm.cdf(self.band_multiplier) - 1.0)

    def table(self) -> pd.DataFrame:
        columns = [
            "Lag", "AC", "PAC", "Q-Stat", "Prob.", "DF",
            "AC Lower", "AC Upper", "PAC Lower", "PAC Upper",
        ]
        data: dict[str, np.ndarray] = {
            "Lag": self.lags,
            "AC": self.ac,
            "PAC": self.pac,
            "Q-Stat": self.q_stat,
            "Prob.": self.pvalues,
            "DF": self.df,
        }
        if self.ac_lower is not None:
            data.update({
                "AC Lower": self.ac_lower,
                "AC Upper": self.ac_upper,
                "PAC Lower": self.pac_lower,
                "PAC Upper": self.pac_upper,
            })
        else:
            columns = columns[:6]
        return pd.DataFrame(data, columns=columns).copy()

    def summary(self) -> str:
        lines = [
            f"Correlogram for {self.series_name}",
            f"Included observations: {self.nobs}",
            f"Excluded missing observations: {self.missing_count}",
            f"Lags: 1 to {self.nlags}; model_df={self.model_df}; alpha={self.alpha:g}",
            f"Bands: +/- {self.band_multiplier:g} SE = +/- {self.band_half_width:.6f} ({self.band_method})",
            "Lag       AC        PAC       Q-Stat       Prob.     DF",
            "------------------------------------------------------------",
        ]
        for row in self.table().itertuples(index=False):
            prob = "NA" if pd.isna(row[4]) else f"{row[4]:.4f}"
            lines.append(f"{int(row[0]):>3d}   {row[1]:>9.4f}  {row[2]:>9.4f}  {row[3]:>11.4f}  {prob:>8}  {int(row[5]):>3d}")
        return "\n".join(lines)

    def interpret(self) -> str:
        valid = np.flatnonzero(np.isfinite(self.pvalues))
        if valid.size == 0:
            return "No Ljung-Box probability is defined because the adjusted degrees of freedom are non-positive."
        idx = int(valid[-1])
        lag = int(self.lags[idx])
        pvalue = float(self.pvalues[idx])
        decision = "rejects" if pvalue < self.alpha else "does not reject"
        return f"At lag {lag}, the Ljung-Box test {decision} the no-autocorrelation null at the {self.alpha:.0%} level (p={pvalue:.4g})."


def ljung_box(ac_values: np.ndarray, *, nobs: int, model_df: int = 0, nlags: int | None = None) -> LjungBoxResult:
    """Compute cumulative Ljung-Box Q statistics from autocorrelations."""
    values = np.asarray(ac_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("ac_values must be one-dimensional")
    if nobs < 2:
        raise ValueError("nobs must be at least 2")
    if not isinstance(model_df, int) or isinstance(model_df, bool) or model_df < 0:
        raise ValueError("model_df must be a non-negative integer")
    if nlags is None:
        nlags = values.size - 1
    if not isinstance(nlags, int) or isinstance(nlags, bool) or nlags < 1:
        raise ValueError("nlags must be a positive integer")
    if nlags >= nobs:
        raise ValueError("nlags must be smaller than nobs")
    if values.size < nlags + 1:
        raise ValueError("ac_values must contain lag 0 through nlags")

    lags = np.arange(1, nlags + 1, dtype=int)
    q_stats = np.empty(nlags, dtype=float)
    df = lags - model_df
    pvalues = np.full(nlags, np.nan, dtype=float)
    cumulative = 0.0
    for i, lag in enumerate(lags):
        cumulative += values[lag] ** 2 / (nobs - lag)
        q_stats[i] = nobs * (nobs + 2) * cumulative
        if df[i] > 0:
            pvalues[i] = float(chi2.sf(q_stats[i], int(df[i])))
    return LjungBoxResult(lags, q_stats, pvalues, df, nobs, model_df)


def correlogram(series, *, nlags: int = 36, model_df: int = 0, alpha: float = 0.05) -> CorrelogramResult:
    """Return a unified EViews-style correlogram result."""
    if not isinstance(model_df, int) or isinstance(model_df, bool) or model_df < 0:
        raise ValueError("model_df must be a non-negative integer")
    ac = acf(series, nlags=nlags, alpha=alpha)
    pc = pacf(series, nlags=nlags, alpha=alpha)
    effective_nlags = min(nlags, ac.nobs - 1)
    lb = ljung_box(ac.values, nobs=ac.nobs, model_df=model_df, nlags=effective_nlags)
    lags = lb.lags
    return CorrelogramResult(
        lags=lags,
        ac=ac.values[lags],
        pac=pc.values[lags],
        q_stat=lb.q_stat,
        pvalues=lb.pvalues,
        df=lb.df,
        nobs=ac.nobs,
        nlags=effective_nlags,
        model_df=model_df,
        alpha=alpha,
        series_name=getattr(ac, "series_name", "series"),
        missing_count=ac.missing_count,
        ac_lower=ac.lower[lags],
        ac_upper=ac.upper[lags],
        pac_lower=pc.lower[lags],
        pac_upper=pc.upper[lags],
        band_multiplier=ac.band_multiplier,
        band_method=ac.band_method,
    )
