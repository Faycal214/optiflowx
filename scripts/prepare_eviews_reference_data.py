#!/usr/bin/env python3
"""Prepare public reference data for Step 13 EViews comparisons.

Requires:
    pip install rdata pandas

The CRAN ARDL package is GPL-3. Its data are fetched into a local
validation directory and are not copied into the MIT StochX source tree.
"""

from __future__ import annotations

import argparse
import io
import tarfile
import urllib.request
from pathlib import Path

CRAN = "https://cran.r-project.org/src/contrib/ARDL_0.2.5.tar.gz"


def download_denmark(out_dir: Path) -> Path:
    import rdata
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = urllib.request.urlopen(CRAN, timeout=60).read()
    rda_path = out_dir / "denmark.rda"
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith("/data/denmark.rda"))
        source = tf.extractfile(member)
        if source is None:
            raise RuntimeError("denmark.rda missing from ARDL archive")
        rda_path.write_bytes(source.read())
    converted = rdata.read_rda(rda_path)
    obj = next(iter(converted.values()))
    csv_path = out_dir / "denmark.csv"
    obj.to_csv(csv_path, index=False)
    return csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("validation_data"))
    args = parser.parse_args()
    print(download_denmark(args.output))


if __name__ == "__main__":
    main()
