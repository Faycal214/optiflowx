#!/usr/bin/env python3
"""Fetch external reference data for local Step 13 validation.

This script deliberately keeps external EViews/R data outside the StochX
repository data tree. The resulting files are local validation artifacts.
"""

from __future__ import annotations

import argparse
import io
import tarfile
import urllib.request
from pathlib import Path


CRAN_ARDL = "https://cran.r-project.org/src/contrib/ARDL_0.2.5.tar.gz"


def fetch_denmark_rda(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    archive = urllib.request.urlopen(CRAN_ARDL, timeout=60).read()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tf:
        member = next(
            m for m in tf.getmembers()
            if m.name.endswith("/data/denmark.rda")
        )
        source = tf.extractfile(member)
        if source is None:
            raise RuntimeError("denmark.rda not found in the CRAN archive")
        output.write_bytes(source.read())
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("validation_data/denmark.rda"))
    args = parser.parse_args()
    path = fetch_denmark_rda(args.output)
    print(path)


if __name__ == "__main__":
    main()
