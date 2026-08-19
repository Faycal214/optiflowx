import json
from pathlib import Path

import numpy as np
import pytest

from stochx.timeseries import CorrelogramResult, correlogram


_FIXTURE = Path(__file__).parent / "fixtures" / "eviews_stage8" / "correlogram_references.json"


def _fixture():
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def test_stage8_11_short_sample_caps_nlags_without_changing_result_contract():
    result = correlogram(np.array([1.0, 2.0, 4.0, 3.0, 5.0]), nlags=20)

    assert isinstance(result, CorrelogramResult)
    assert result.nobs == 5
    assert result.nlags == 4
    assert result.lags.tolist() == [1, 2, 3, 4]
    assert result.AC.shape == (4,)
    assert result.PAC.shape == (4,)
    assert result.Q_Stat.shape == (4,)
    assert result.Prob.shape == (4,)
    assert result.DF.tolist() == [1, 2, 3, 4]
    expected_band = 2.0 / np.sqrt(5.0)
    np.testing.assert_allclose(result.ac_lower, -expected_band)
    np.testing.assert_allclose(result.ac_upper, expected_band)
    np.testing.assert_allclose(result.pac_lower, -expected_band)
    np.testing.assert_allclose(result.pac_upper, expected_band)


def test_stage8_11_missing_values_use_one_common_effective_sample():
    raw = np.array([1.0, 4.0, np.nan, 2.0, 7.0, np.nan, 5.0, 9.0])
    cleaned = raw[~np.isnan(raw)]

    result = correlogram(raw, nlags=6)
    reference = correlogram(cleaned, nlags=6)

    assert result.nobs == len(cleaned)
    assert result.missing_count == 2
    assert result.nlags == reference.nlags
    np.testing.assert_array_equal(result.lags, reference.lags)
    np.testing.assert_allclose(result.AC, reference.AC, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.PAC, reference.PAC, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.Q_Stat, reference.Q_Stat, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.Prob, reference.Prob, rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(result.ac_lower, reference.ac_lower, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.ac_upper, reference.ac_upper, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.pac_lower, reference.pac_lower, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.pac_upper, reference.pac_upper, rtol=1e-12, atol=1e-12)


def test_stage8_11_missing_values_do_not_change_frozen_table_schema():
    result = correlogram(np.array([1.0, np.nan, 2.0, 3.0, np.nan, 5.0]), nlags=4)
    assert list(result.table().columns) == [
        "Lag", "AC", "PAC", "Q-Stat", "Prob.", "DF",
        "AC Lower", "AC Upper", "PAC Lower", "PAC Upper",
    ]
    assert result.table().shape == (4, 10)


def test_stage8_11_nonpositive_residual_degrees_of_freedom_are_undefined():
    data = _fixture()
    case = data["residual"]
    settings = data["settings"]["residual"]
    result = correlogram(
        np.asarray(case["raw_series"], dtype=float),
        nlags=settings["nlags"],
        model_df=settings["model_df"],
        alpha=settings["alpha"],
    )

    assert result.DF[:2].tolist() == [-1, 0]
    assert np.isnan(result.Prob[:2]).all()
    assert np.isfinite(result.Prob[2:]).all()
    expected = np.asarray([np.nan, np.nan, *case["expected"]["Prob."][2:]], dtype=float)
    np.testing.assert_allclose(result.Prob, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_stage8_11_custom_alpha_changes_metadata_and_decision_level_not_numerical_fields():
    data = _fixture()
    raw = np.asarray(data["ordinary"]["raw_series"], dtype=float)
    baseline = correlogram(raw, nlags=6, model_df=0, alpha=0.05)
    alternate = correlogram(raw, nlags=6, model_df=0, alpha=0.10)

    assert baseline.alpha == 0.05
    assert alternate.alpha == 0.10
    np.testing.assert_allclose(alternate.AC, baseline.AC, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(alternate.PAC, baseline.PAC, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(alternate.Q_Stat, baseline.Q_Stat, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(alternate.Prob, baseline.Prob, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(alternate.DF, baseline.DF)
    np.testing.assert_allclose(alternate.ac_lower, baseline.ac_lower, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(alternate.ac_upper, baseline.ac_upper, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(alternate.pac_lower, baseline.pac_lower, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(alternate.pac_upper, baseline.pac_upper, rtol=1e-12, atol=1e-12)


def test_stage8_11_fixture_parity_is_immutable_after_projection_and_comparison():
    data = _fixture()
    case = data["ordinary"]
    result = correlogram(np.asarray(case["raw_series"], dtype=float), nlags=6, model_df=0, alpha=0.05)
    before = {
        name: value.copy()
        for name, value in {
            "AC": result.AC,
            "PAC": result.PAC,
            "Q_Stat": result.Q_Stat,
            "Prob": result.Prob,
            "DF": result.DF,
            "ac_lower": result.ac_lower,
            "ac_upper": result.ac_upper,
            "pac_lower": result.pac_lower,
            "pac_upper": result.pac_upper,
        }.items()
    }

    _ = result.table()
    _ = result.summary()
    _ = result.interpret()

    for name, values in before.items():
        current = getattr(result, name)
        np.testing.assert_array_equal(current, values)
        assert not current.flags.writeable

    assert result.AC is result.ac
    assert result.PAC is result.pac
    assert result.Q_Stat is result.q_stat
    assert result.QStat is result.q_stat
    assert result.Prob is result.pvalues
    assert result.PValues is result.pvalues
    assert result.DF is result.df


def test_stage8_11_invalid_missing_and_short_inputs_fail_cleanly():
    with pytest.raises(ValueError, match="at least two non-missing"):
        correlogram(np.array([np.nan, np.nan]), nlags=2)
    with pytest.raises(ValueError, match="at least two non-missing"):
        correlogram(np.array([1.0, np.nan]), nlags=1)
    with pytest.raises(ValueError, match="must not contain infinite"):
        correlogram(np.array([1.0, np.inf, 2.0]), nlags=1)
