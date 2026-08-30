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


def test_public_eviews_diagnostic_capture_contains_verified_bg_numbers():
    fixture = load_fixture(str(BASE / "eviews_diagnostics_reference.json"))
    stats = fixture["views"]["serial_correlation_lm"]["statistics"]
    assert stats["F-statistic"] == 23.02572
    assert stats["Obs*R-squared"] == 43.78175
    assert stats["df"] == 2
    assert fixture["equation"]["nobs"] == 734


def test_public_eviews_cointegration_capture_contains_verified_johansen_numbers():
    fixture = load_fixture(str(BASE / "eviews_cointegration_reference.json"))
    trace = fixture["system"]["rank_tests"]["trace"]
    maxeig = fixture["system"]["rank_tests"]["max_eigen"]
    assert trace[0]["eigenvalue"] == 0.433165
    assert trace[0]["statistic"] == 49.14436
    assert maxeig[0]["statistic"] == 30.08745
    assert maxeig[0]["critical_5"] == 28.58808
    assert fixture["system"]["selected_rank"]["trace_5"] == 0
    assert fixture["system"]["selected_rank"]["max_eigen_5"] == 1


def test_public_eviews_hansen_capture_is_recorded_without_claiming_stochx_parity():
    fixture = load_fixture(str(BASE / "eviews_cointegration_reference.json"))
    assert fixture["single_equation"]["tests"]["hansen"]["statistic"] == 0.5755
    assert fixture["parity"]["status"] == "partially_populated"
