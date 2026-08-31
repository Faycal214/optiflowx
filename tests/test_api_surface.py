"""Contract tests for the canonical time-series-only StochX public API."""

import numpy as np

from stochx.timeseries import TimeSeries, Workfile


def test_canonical_series_api():
    series = TimeSeries([1.0, 2.0, 3.0], name="Y")
    assert len(series) == 3
    assert series.name == "Y"


def test_canonical_workfile_api():
    wf = Workfile()
    wf.add("Y", np.array([1.0, 2.0, 3.0]))
    assert "Y" in wf.names
    assert len(wf.get("Y")) == 3


def test_canonical_equation_api():
    wf = Workfile()
    wf.add("Y", np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    wf.add("X", np.array([2.0, 3.0, 4.0, 5.0, 6.0]))
    eq = wf.ls("Y C X", name="EQ")
    assert eq.name == "EQ"
    assert hasattr(eq, "forecast")
