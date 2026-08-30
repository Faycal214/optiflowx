import json
from pathlib import Path

from stochx.timeseries import validate_fixture_schema


FIXTURE = Path(__file__).parent / "fixtures" / "eviews_arima_forecast_reference.json"


def test_eviews_arima_forecast_reference_fixture_is_valid():
    reference = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert validate_fixture_schema({
        "schema_version": 1,
        "models": {
            "ARIMA_REFERENCE": {
                "category": "arima",
                "specification": reference["expected_selection"],
                "nobs": None,
                "coefficients": {},
                "statistics": {},
                "roots": {"ar": [], "ma": [], "sar": [], "sma": []},
            }
        },
    }) == []


def test_documented_eviews_electricity_case_selection_metadata():
    reference = json.loads(FIXTURE.read_text(encoding="utf-8"))
    spec = reference["specification"]
    selected = reference["expected_selection"]
    assert spec["candidate_models_expected"] == 100
    assert spec["max_ar"] == 4
    assert spec["max_ma"] == 4
    assert spec["max_sar"] == 1
    assert spec["max_sma"] == 1
    assert spec["periods"] == 12
    assert spec["selection"] == "AIC"
    assert spec["estimation_sample"] == "2005M01 2014M04"
    assert spec["forecast_sample"] == "2014M05 2015M12"
    assert selected == {"transformation": "log", "d": 1, "p": 3, "q": 3, "P": 1, "D": 0, "Q": 1, "periods": 12}


def test_forecast_reference_explicitly_marks_unpublished_numeric_vectors_as_pending():
    reference = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert reference["forecast"]["status"] == "reference_values_not_published_in_text"
    assert reference["source"]["numeric_forecast_capture"] == "pending"
