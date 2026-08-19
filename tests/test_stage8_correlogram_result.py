import numpy as np
import pandas as pd
import pytest

from stochx.timeseries import CorrelogramResult, arma, correlogram, white_noise


def test_stage8_6_ordinary_result_public_contract_and_aliases():
    result = correlogram(white_noise(120, rng=21), nlags=8, model_df=0, alpha=0.05)

    assert isinstance(result, CorrelogramResult)
    assert result.nobs == 120
    assert result.nlags == 8
    assert result.model_df == 0
    assert result.alpha == 0.05
    assert result.series_name == "WN"
    assert result.lags.tolist() == list(range(1, 9))

    assert np.array_equal(result.AC, result.ac)
    assert np.array_equal(result.PAC, result.pac)
    assert np.array_equal(result.DF, result.df)
    assert np.array_equal(result.Q_Stat, result.q_stat)
    assert np.array_equal(result.QStat, result.q_stat)
    assert np.array_equal(result.Prob, result.pvalues)
    assert np.array_equal(result.PValues, result.pvalues)

    assert result.ac_lower.shape == result.lags.shape
    assert result.ac_upper.shape == result.lags.shape
    assert result.pac_lower.shape == result.lags.shape
    assert result.pac_upper.shape == result.lags.shape


def test_stage8_6_residual_result_preserves_model_df_and_band_metadata():
    result = correlogram(arma(p=1, q=1, phi=[0.45], theta=[0.25], n=160, rng=7), nlags=7, model_df=2)

    assert result.nobs == 160
    assert result.nlags == 7
    assert result.model_df == 2
    assert np.array_equal(result.DF, np.arange(1, 8) - 2)
    assert np.isnan(result.Prob[:2]).all()
    assert np.isfinite(result.Prob[2:]).all()
    assert result.band_method == "approx_two_standard_errors"
    assert result.band_multiplier == 2.0
    expected_half_width = 2.0 / np.sqrt(result.nobs)
    assert np.isclose(result.band_half_width, expected_half_width)


def test_stage8_6_table_is_stable_display_projection():
    result = correlogram(white_noise(90, rng=4), nlags=5, model_df=0)
    table = result.table()

    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == [
        "Lag", "AC", "PAC", "Q-Stat", "Prob.", "DF",
        "AC Lower", "AC Upper", "PAC Lower", "PAC Upper",
    ]
    assert np.array_equal(table["Lag"].to_numpy(), result.lags)
    assert np.allclose(table["AC"].to_numpy(), result.AC)
    assert np.allclose(table["PAC"].to_numpy(), result.PAC)
    assert np.allclose(table["Q-Stat"].to_numpy(), result.Q_Stat)
    assert np.allclose(table["Prob."].to_numpy(), result.Prob, equal_nan=True)
    assert np.array_equal(table["DF"].to_numpy(), result.DF)
    assert np.allclose(table["AC Lower"].to_numpy(), result.ac_lower)
    assert np.allclose(table["AC Upper"].to_numpy(), result.ac_upper)
    assert np.allclose(table["PAC Lower"].to_numpy(), result.pac_lower)
    assert np.allclose(table["PAC Upper"].to_numpy(), result.pac_upper)


def test_stage8_6_constructor_rejects_inconsistent_metadata_and_shapes():
    base = correlogram(white_noise(50, rng=3), nlags=4)
    kwargs = dict(
        lags=base.lags,
        ac=base.ac,
        pac=base.pac,
        q_stat=base.q_stat,
        pvalues=base.pvalues,
        df=base.df,
        nobs=base.nobs,
        nlags=base.nlags,
        model_df=base.model_df,
        alpha=base.alpha,
        series_name=base.series_name,
        missing_count=base.missing_count,
        ac_lower=base.ac_lower,
        ac_upper=base.ac_upper,
        pac_lower=base.pac_lower,
        pac_upper=base.pac_upper,
        band_multiplier=base.band_multiplier,
        band_method=base.band_method,
    )

    bad = dict(kwargs, nlags=5)
    with pytest.raises(ValueError, match="length nlags"):
        CorrelogramResult(**bad)

    bad = dict(kwargs, lags=np.array([1, 2, 4, 5]))
    with pytest.raises(ValueError, match="lags must be exactly"):
        CorrelogramResult(**bad)

    bad = dict(kwargs, df=np.array([1, 2, 3, 9]))
    with pytest.raises(ValueError, match="DF must equal"):
        CorrelogramResult(**bad)

    bad = dict(kwargs, ac_upper=None)
    with pytest.raises(ValueError, match="all AC/PAC lower and upper"):
        CorrelogramResult(**bad)

    bad = dict(kwargs, alpha=1.0)
    with pytest.raises(ValueError, match="alpha"):
        CorrelogramResult(**bad)


def test_stage8_6_arrays_are_read_only_and_cannot_mutate_result():
    result = correlogram(white_noise(80, rng=9), nlags=5)

    with pytest.raises(ValueError):
        result.ac[0] = 999.0
    with pytest.raises(ValueError):
        result.Q_Stat[0] = 999.0
    with pytest.raises(ValueError):
        result.ac_lower[0] = 999.0

    assert result.AC[0] == result.ac[0]
    assert np.isfinite(result.Q_Stat).all()
