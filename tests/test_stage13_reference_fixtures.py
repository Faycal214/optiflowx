import json
from pathlib import Path

from stochx.timeseries.parity import (
    load_fixture,
    validate_cointegration_reference,
    validate_diagnostics_reference,
)


BASE = Path(__file__).parent / "fixtures"


def test_diagnostics_reference_fixture_has_all_eviews_view_families():
    fixture = load_fixture(str(BASE / "eviews_diagnostics_reference.json"))
    report = validate_diagnostics_reference(fixture)
    assert report.passed, report.text()


def test_cointegration_reference_fixture_has_single_equation_and_system_families():
    fixture = load_fixture(str(BASE / "eviews_cointegration_reference.json"))
    report = validate_cointegration_reference(fixture)
    assert report.passed, report.text()


def test_cointegration_fixture_explicitly_tracks_unimplemented_hansen_and_park():
    fixture = load_fixture(str(BASE / "eviews_cointegration_reference.json"))
    tests = fixture["single_equation"]["tests"]
    assert tests["hansen"]["method"] == "hansen"
    assert tests["park"]["method"] == "park"
    assert fixture["parity"]["numeric_capture_pending" if "numeric_capture_pending" in fixture["parity"] else "status"] == "numeric_capture_pending"


def test_diagnostics_fixture_preserves_eviews_arma_availability_metadata():
    fixture = load_fixture(str(BASE / "eviews_diagnostics_reference.json"))
    assert fixture["equation"]["arma_terms"] == 0
    assert fixture["views"]["stability"]["available_for_equation"] is True
