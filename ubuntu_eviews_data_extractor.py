#!/usr/bin/env python3
"""Prepare public validation-data templates without fabricating observations.

This script never creates random econometric data. It either reads a supplied
CSV or downloads the public Denmark source into a local validation directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


REQUIRED_DENMARK = ["LRM", "LRY", "LPY", "IBO", "IDE"]


def add_seasonal_dummies(frame: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    out = frame.copy()
    if date_column not in out:
        raise ValueError(f"missing date column: {date_column}")
    dates = out[date_column].astype(str)
    quarter = dates.str.extract(r"Q([1-4])", expand=False)
    if quarter.isna().any():
        raise ValueError("date column must contain quarterly labels such as 1974Q1")
    q = quarter.astype(int)
    out["D1"] = (q == 1).astype(int)
    out["D2"] = (q == 2).astype(int)
    out["D3"] = (q == 3).astype(int)
    return out


def validate_denmark(frame: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_DENMARK if c not in frame.columns]
    if missing:
        raise ValueError(f"missing Denmark columns: {missing}")
    if len(frame) < 55:
        raise ValueError("Denmark reference dataset appears incomplete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="CSV containing the real public/reference dataset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--denmark", action="store_true")
    parser.add_argument("--date-column", default="date")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    if args.denmark:
        validate_denmark(frame)
        if not {"D1", "D2", "D3"}.issubset(frame.columns):
            frame = add_seasonal_dummies(frame, args.date_column)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(frame)} observations)")


if __name__ == "__main__":
    main()
