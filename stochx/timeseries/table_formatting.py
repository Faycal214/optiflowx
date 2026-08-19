"""Deterministic EViews-style presentation helpers for time-series results."""

from __future__ import annotations

import pandas as pd

from .correlogram import CorrelogramResult


def format_correlogram_table(
    result: CorrelogramResult,
    *,
    precision: int = 4,
    missing: str = "NA",
) -> pd.DataFrame:
    """Return a string-valued EViews-style correlogram table.

    This is a presentation-only projection. The numerical result returned by
    ``CorrelogramResult.table()`` and all result arrays are left untouched.
    """
    if not isinstance(result, CorrelogramResult):
        raise TypeError("result must be a CorrelogramResult")
    if not isinstance(precision, int) or isinstance(precision, bool) or precision < 0:
        raise ValueError("precision must be a non-negative integer")
    if not isinstance(missing, str) or missing == "":
        raise ValueError("missing must be a non-empty string")

    numeric = result.table()
    formatted = pd.DataFrame(index=numeric.index)
    integer_columns = {"Lag", "DF"}

    for column in numeric.columns:
        series = numeric[column]
        if column in integer_columns:
            formatted[column] = series.astype(int).astype(str)
        else:
            formatted[column] = series.map(
                lambda value: missing if pd.isna(value) else f"{float(value):.{precision}f}"
            )

    return formatted.loc[:, list(numeric.columns)]


def format_correlogram(
    result: CorrelogramResult,
    *,
    precision: int = 4,
    missing: str = "NA",
    include_header: bool = True,
) -> str:
    """Render a deterministic fixed-width EViews-style correlogram."""
    table = format_correlogram_table(result, precision=precision, missing=missing)
    return table.to_string(index=False, header=include_header)
