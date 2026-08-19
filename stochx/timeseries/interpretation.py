"""Interpretation helpers for EViews-style time-series diagnostics."""

from __future__ import annotations

import numpy as np

from .correlogram import CorrelogramResult


def interpret_correlogram(
    result: CorrelogramResult,
    *,
    alpha: float | None = None,
    max_spikes: int | None = None,
) -> str:
    """Interpret AC/PAC spikes and the Ljung-Box result without changing data.

    ``result`` remains the single source of truth for all numerical decisions.
    The function only reads its frozen arrays and metadata.
    """
    if not isinstance(result, CorrelogramResult):
        raise TypeError("result must be a CorrelogramResult")
    level = result.alpha if alpha is None else float(alpha)
    if not 0 < level < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if max_spikes is not None and (
        not isinstance(max_spikes, int) or isinstance(max_spikes, bool) or max_spikes < 1
    ):
        raise ValueError("max_spikes must be a positive integer or None")

    def _significant(values: np.ndarray, lower: np.ndarray | None, upper: np.ndarray | None) -> list[int]:
        if lower is None or upper is None:
            return []
        mask = (values < lower) | (values > upper)
        lags = result.lags[mask].astype(int).tolist()
        return lags if max_spikes is None else lags[:max_spikes]

    ac_spikes = _significant(result.ac, result.ac_lower, result.ac_upper)
    pac_spikes = _significant(result.pac, result.pac_lower, result.pac_upper)

    parts: list[str] = []
    kind = "residual" if result.model_df > 0 else "ordinary"
    parts.append(f"{kind.capitalize()} correlogram for {result.series_name} ({result.nobs} observations).")

    if ac_spikes:
        parts.append("Significant AC spikes at lag(s) " + ", ".join(map(str, ac_spikes)) + ".")
    else:
        parts.append("No AC spikes are outside the displayed confidence bands.")

    if pac_spikes:
        parts.append("Significant PAC spikes at lag(s) " + ", ".join(map(str, pac_spikes)) + ".")
    else:
        parts.append("No PAC spikes are outside the displayed confidence bands.")

    valid = np.flatnonzero(np.isfinite(result.Prob))
    if valid.size == 0:
        parts.append("The Ljung-Box probability is undefined for all displayed lags because the adjusted degrees of freedom are non-positive.")
        return " ".join(parts)

    idx = int(valid[-1])
    lag = int(result.lags[idx])
    q = float(result.Q_Stat[idx])
    pvalue = float(result.Prob[idx])
    decision = "rejects" if pvalue < level else "does not reject"
    parts.append(
        f"At lag {lag}, Ljung-Box Q={q:.4f} with df={int(result.DF[idx])} and p={pvalue:.4g}; "
        f"the test {decision} the no-autocorrelation null at the {level:.0%} level."
    )

    if result.model_df > 0:
        parts.append(f"The Ljung-Box degrees of freedom are adjusted by model_df={result.model_df}.")

    return " ".join(parts)
