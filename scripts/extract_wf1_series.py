#!/usr/bin/env python3
"""Extract standard numeric series from a legacy EViews WF1 file.

This is a validation utility, not a package runtime dependency. It supports
the standard uncompressed series records used by the supplied Uroot.WF1.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import pandas as pd


NA = 1e-37
SERIES_TYPE = 44
STANDARD_STORAGE = 11
RECORD_SIZE = 70


def read_workfile(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    raw = path.read_bytes()
    header_size = struct.unpack_from("<I", raw, 80)[0]
    nvars = struct.unpack_from("<i", raw, 114)[0]
    frequency = struct.unpack_from("<h", raw, 124)[0]
    start_year = struct.unpack_from("<i", raw, 128)[0]
    start_period = struct.unpack_from("<h", raw, 132)[0]
    nobs = struct.unpack_from("<i", raw, 140)[0]

    if frequency not in {1, 4, 12}:
        raise ValueError(f"unsupported frequency code: {frequency}")

    labels = []
    for i in range(nobs):
        period = start_period - 1 + i
        year = start_year + period // frequency
        subperiod = period % frequency + 1
        if frequency == 4:
            labels.append(f"{year}Q{subperiod}")
        elif frequency == 12:
            labels.append(f"{year}M{subperiod:02d}")
        else:
            labels.append(str(year))

    records = {}
    records_offset = header_size + 26
    for i in range(nvars):
        off = records_offset + i * RECORD_SIZE
        storage = struct.unpack_from("<h", raw, off + 4)[0]
        pointer = struct.unpack_from("<I", raw, off + 14)[0]
        name = raw[off + 22:off + 54].split(b"\0", 1)[0].decode("latin1")
        obj_type = struct.unpack_from("<h", raw, off + 62)[0]

        if obj_type != SERIES_TYPE or storage != STANDARD_STORAGE or not name or pointer == 0:
            continue

        count = struct.unpack_from("<i", raw, pointer)[0]
        values = np.frombuffer(
            raw,
            dtype="<f8",
            count=count,
            offset=pointer + 22,
        ).copy()
        values[np.abs(values - NA) < 1e-36] = np.nan
        records[name] = values

    return labels, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workfile", type=Path)
    parser.add_argument("--series", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dates, series = read_workfile(args.workfile)
    missing = [name for name in args.series if name not in series]
    if missing:
        raise SystemExit(f"missing series: {', '.join(missing)}")

    frame = pd.DataFrame({name: series[name] for name in args.series})
    frame.insert(0, "date", dates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(frame)} observations)")


if __name__ == "__main__":
    main()
