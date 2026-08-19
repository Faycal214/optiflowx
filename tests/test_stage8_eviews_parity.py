import json
from pathlib import Path

import numpy as np

from stochx.timeseries import correlogram


_FIXTURE = Path(__file__).parent / "fixtures" / "eviews_stage8" / "correlogram_references.json"


def _load_fixture():
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _assert_case(case):
    result = correlogram(
        np.asarray(case["raw_series"], dtype=float),
        nlags=6,
        model_df=case["model_df"],
        alpha=case["alpha"],
    )
    expected = case["expected"]

    assert result.nobs == case["nobs"]
    assert result.missing_count == case["missing_count"]
    assert result.nlags == 6
    assert result.model_df == case["model_df"]
    assert result.alpha == case["alpha"]

    np.testing.assert_array_equal(result.lags, np.asarray(expected["lags"], dtype=int))
    np.testing.assert_allclose(result.AC, expected["AC"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.PAC, expected["PAC"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.Q_Stat, expected["Q-Stat"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        result.Prob,
        np.asarray([np.nan if x is None else x for x in expected["Prob."]], dtype=float),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    np.testing.assert_array_equal(result.DF, np.asarray(expected["DF"], dtype=int))
    np.testing.assert_allclose(result.ac_lower, expected["AC Lower"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.ac_upper, expected["AC Upper"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.pac_lower, expected["PAC Lower"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.pac_upper, expected["PAC Upper"], rtol=1e-12, atol=1e-12)


def test_stage8_10_ordinary_eviews_reference_parity():
    data = _load_fixture()
    _assert_case({**data["ordinary"], **data["settings"]["ordinary"]})


def test_stage8_10_residual_eviews_reference_parity():
    data = _load_fixture()
    _assert_case({**data["residual"], **data["settings"]["residual"]})


def test_stage8_10_fixture_metadata_freezes_reference_conventions():
    data = _load_fixture()
    assert data["schema_version"] == "8.10"
    assert data["conventions"]["ac_method"] == "eviews_common_overall_mean"
    assert data["conventions"]["pac_method"] == "recursive_box_jenkins_durbin_levinson"
    assert data["conventions"]["q_stat_method"] == "ljung_box"
    assert data["conventions"]["df_formula"] == "lag-model_df"
    assert data["conventions"]["nonpositive_df_probability"] == "NaN"
    assert data["conventions"]["band_formula"] == "plus_minus_2/sqrt(nobs)"


def test_stage8_10_eviews_reference_does_not_mutate_frozen_result_contract():
    data = _load_fixture()
    case = {**data["ordinary"], **data["settings"]["ordinary"]}
    result = correlogram(np.asarray(case["raw_series"], dtype=float), nlags=6)
    before = result.table().copy(deep=True)

    _assert_case(case)

    assert result.AC is result.ac
    assert result.Q_Stat is result.q_stat
    assert result.Prob is result.pvalues
    assert result.DF is result.df
    assert result.table().equals(before)
