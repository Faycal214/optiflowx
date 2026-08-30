import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stochx.timeseries import Workfile, compare_forecast, compare_number


DATA = Path("validation_data")


def test_denmark_reference_execution_is_opt_in_and_same_sample():
    csv = DATA / "denmark.csv"
    if not csv.exists():
        pytest.skip("run scripts/prepare_eviews_reference_data.py first")
    frame = pd.read_csv(csv)
    assert {"LRM", "LRY", "IBO", "IDE"} <= set(frame.columns)
    frame = frame.iloc[2:].reset_index(drop=True)
    assert len(frame) == 53

    wf = Workfile(frequency="Q")
    wf.add("LRM", frame["LRM"].to_numpy())
    wf.add("LRY", frame["LRY"].to_numpy())
    wf.add("IBO", frame["IBO"].to_numpy())
    wf.add("IDE", frame["IDE"].to_numpy())

    johansen = wf.johansen(["LRM", "LRY", "IBO", "IDE"], k_ar_diff=1, det_order=0)
    assert johansen.rank >= 0
    assert johansen.rank <= 4


@pytest.mark.skipif(
    not os.environ.get("STOCHX_EVIEWS_CAPTURE_DIR"),
    reason="requires a local capture exported from EViews",
)
def test_eviews_numeric_capture_can_be_compared():
    capture_dir = Path(os.environ["STOCHX_EVIEWS_CAPTURE_DIR"])
    assert capture_dir.exists()
