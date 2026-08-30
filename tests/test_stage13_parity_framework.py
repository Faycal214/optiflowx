import json
from pathlib import Path

import numpy as np

from stochx.timeseries.parity import (
    ParityReport,
    compare_array,
    compare_dataframe,
    compare_equation_fixture,
    compare_number,
    display_tolerance,
)


def test_display_tolerance_matches_reference_precision():
    assert display_tolerance("2.569260") == 0.0000005
    assert display_tolerance("0.94") == 0.005


def test_number_comparison_uses_fixture_precision():
    result = compare_number(2.5692601, "2.569260", name="AIC")
    assert result.passed
    assert result.abs_error is not None


def test_parity_report_exposes_failures_and_summary():
    report = ParityReport("EQ18")
    report.comparisons.append(compare_number(1.0, "1.000000", name="coefficient.C"))
    assert report.passed
    assert report.summary()["failures"] == 0
    assert "[PASS] EQ18" in report.text()


def test_array_comparison_checks_lengths_and_values():
    report = ParityReport("forecast")
    compare_array([1.0, 2.0], ["1.0", "2.0"], report=report, name="Forecast")
    assert report.passed


def test_dataframe_comparison_requires_same_labels():
    import pandas as pd
    actual = pd.DataFrame({"A": [1.0, 2.0]}, index=["x", "y"])
    expected = pd.DataFrame({"A": [1.0, 2.0]}, index=["x", "y"])
    report = compare_dataframe(actual, expected, model_name="table")
    assert report.passed


def test_fixture_schema_contains_all_published_phase_b_equations():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "eviews_phase_b_expected.json").read_text()
    )
    assert set(fixture["equations"]) == {"EQ18", "EQ19", "EQ20", "EQ21"}
    for reference in fixture["equations"].values():
        assert reference["nobs"] == 624
        assert "coefficients" in reference
        assert "statistics" in reference
        assert "roots" in reference
