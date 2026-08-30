import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stochx.timeseries import Workfile, compare_johansen_reference


FIXTURE = Path(__file__).parent / "fixtures" / "eviews_cointegration_reference.json"
DATA = Path("validation_data") / "denmark.csv"


def test_denmark_johansen_can_execute_against_eviews_reference_when_dataset_is_prepared():
    if not DATA.exists():
        pytest.skip("prepare validation_data/denmark.csv with scripts/prepare_eviews_reference_data.py")
    reference = json.loads(FIXTURE.read_text(encoding="utf-8"))
    frame = pd.read_csv(DATA)

    # EViews example: LRM, LRY, IBO, IDE; 1974Q3–1987Q3.
    required = ["LRM", "LRY", "IBO", "IDE"]
    assert set(required) <= set(frame.columns)
    frame = frame[required].iloc[2:].reset_index(drop=True)
    assert len(frame) == 53

    wf = Workfile(frequency="Q")
    for name in required:
        wf.add(name, frame[name].to_numpy(dtype=float))

    result = wf.johansen(required, k_ar_diff=1, det_order=0)
    report = compare_johansen_reference(result, reference, rtol=1e-4)

    # This assertion is intentionally strict: if it fails, investigate the
    # deterministic specification/sample/Johansen implementation before
    # changing tolerances.
    assert report.passed, report.text()
