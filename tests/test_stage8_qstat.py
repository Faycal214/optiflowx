import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from stochx.timeseries import CorrelogramResult, TimeSeries, correlogram
from stochx.timeseries.correlogram import ljung_box
from stochx.timeseries.correlation import acf


def _ar1(phi: float, n: int, seed: int) -> TimeSeries:
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=n)
    values = np.empty(n)
    values[0] = eps[0]
    for t in range(1, n):
        values[t] = phi * values[t - 1] + eps[t]
    return TimeSeries(values, name="AR1")


def _arma11(phi: float, theta: float, n: int, seed: int) -> TimeSeries:
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=n)
    values = np.empty(n)
    values[0] = eps[0]
    for t in range(1, n):
        values[t] = phi * values[t - 1] + eps[t] + theta * eps[t - 1]
    return TimeSeries(values, name="ARMA11")


def _white_noise(n: int, seed: int) -> TimeSeries:
    return TimeSeries(np.random.default_rng(seed).normal(size=n), name="WN")


def _reference_q(ac_values, nobs, nlags):
    """Independently reconstruct Ljung-Box Q from displayed lag-1..lag-K AC values."""
    ac_values = np.asarray(ac_values, dtype=float)
    if ac_values.size == nlags + 1:
        lag_values = ac_values[1 : nlags + 1]
    elif ac_values.size == nlags:
        lag_values = ac_values
    else:
        raise ValueError("ac_values must contain either lag 0..K or lag 1..K")

    q = []
    cumulative = 0.0
    for lag, rho in enumerate(lag_values, start=1):
        cumulative += rho**2 / (nobs - lag)
        q.append(nobs * (nobs + 2) * cumulative)
    return np.asarray(q)


def test_ljung_box_matches_independent_formula_for_white_noise():
    y = _white_noise(250, 123)
    ac = acf(y, nlags=12)
    result = ljung_box(ac.values, nobs=ac.nobs, nlags=12)
    expected_q = _reference_q(ac.values, ac.nobs, 12)
    expected_p = np.asarray([chi2.sf(expected_q[k - 1], k) for k in range(1, 13)])

    assert np.allclose(result.q_stat, expected_q, rtol=1e-12, atol=1e-12)
    assert np.allclose(result.pvalues, expected_p, rtol=1e-12, atol=1e-12)
    assert np.array_equal(result.df, np.arange(1, 13))
    assert result.nobs == 250
    assert result.model_df == 0


def test_correlogram_unified_result_exposes_auditable_fields_for_ordinary_series():
    y = _white_noise(150, 7)
    result = correlogram(y, nlags=10, model_df=0, alpha=0.05)

    assert isinstance(result, CorrelogramResult)
    assert result.nobs == 150
    assert result.nlags == 10
    assert result.model_df == 0
    assert result.alpha == 0.05
    assert result.lags.tolist() == list(range(1, 11))
    assert result.DF.tolist() == list(range(1, 11))
    assert np.allclose(result.Q_Stat, result.q_stat)
    assert np.allclose(result.Prob, result.pvalues, equal_nan=True)
    assert np.isfinite(result.Q_Stat).all()
    assert np.isfinite(result.Prob).all()

    table = result.table()
    assert isinstance(table, pd.DataFrame)
    assert list(table["Lag"]) == list(range(1, 11))
    assert {"AC", "PAC", "Q-Stat", "Prob.", "DF"}.issubset(table.columns)


def test_residual_correlogram_exposes_adjusted_df_and_probabilities():
    y = _arma11(0.45, 0.25, 300, 42)
    result = correlogram(y, nlags=8, model_df=2, alpha=0.05)

    assert result.nobs == 300
    assert result.nlags == 8
    assert result.model_df == 2
    assert np.array_equal(result.DF, np.arange(1, 9) - 2)
    assert np.isnan(result.Prob[:2]).all()
    assert np.isfinite(result.Prob[2:]).all()

    expected_q = _reference_q(result.ac, result.nobs, 8)
    assert np.allclose(result.Q_Stat, expected_q, rtol=1e-12, atol=1e-12)
    for i, df in enumerate(result.DF):
        if df > 0:
            assert np.isclose(result.Prob[i], chi2.sf(expected_q[i], int(df)), rtol=1e-12, atol=1e-12)


def test_result_table_is_the_display_projection_of_auditable_arrays():
    result = correlogram(_white_noise(100, 11), nlags=6, model_df=1, alpha=0.10)
    table = result.table()

    assert np.array_equal(table["Lag"].to_numpy(), result.lags)
    assert np.allclose(table["AC"].to_numpy(), result.ac)
    assert np.allclose(table["PAC"].to_numpy(), result.pac)
    assert np.allclose(table["Q-Stat"].to_numpy(), result.Q_Stat)
    assert np.allclose(table["Prob."].to_numpy(), result.Prob, equal_nan=True)
    assert np.array_equal(table["DF"].to_numpy(), result.DF)


def test_correlogram_metadata_and_q_arrays_are_consistent():
    result = correlogram(_ar1(0.65, 220, 19), nlags=12, model_df=0)
    assert result.lags.shape == (result.nlags,)
    assert result.ac.shape == result.lags.shape
    assert result.pac.shape == result.lags.shape
    assert result.Q_Stat.shape == result.lags.shape
    assert result.Prob.shape == result.lags.shape
    assert result.DF.shape == result.lags.shape
    assert np.all(np.diff(result.Q_Stat) >= -1e-12)


def test_stage8_5_ordinary_correlogram_exposes_eviews_bands():
    result = correlogram(white_noise(150, rng=7), nlags=10, model_df=0, alpha=0.05)
    expected = 2.0 / np.sqrt(150)

    assert result.band_method == "approx_two_standard_errors"
    assert result.band_multiplier == 2.0
    assert np.isclose(result.band_standard_error, 1.0 / np.sqrt(150))
    assert np.isclose(result.band_half_width, expected)
    assert np.isclose(result.band_confidence_level, 2.0 * 0.9772498680518208 - 1.0)
    np.testing.assert_allclose(result.ac_lower, -expected, rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(result.ac_upper, expected, rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(result.pac_lower, -expected, rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(result.pac_upper, expected, rtol=1e-15, atol=1e-15)


def test_stage8_5_residual_correlogram_bands_use_shared_effective_nobs_and_ignore_model_df():
    result = correlogram(arma(p=1, q=1, phi=[0.45], theta=[0.25], n=300, rng=42), nlags=8, model_df=2, alpha=0.05)
    expected = 2.0 / np.sqrt(300)

    assert result.nobs == 300
    assert result.model_df == 2
    assert result.band_multiplier == 2.0
    assert np.isclose(result.band_half_width, expected)
    np.testing.assert_allclose(result.ac_lower, -expected, rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(result.ac_upper, expected, rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(result.pac_lower, -expected, rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(result.pac_upper, expected, rtol=1e-15, atol=1e-15)


def test_model_df_must_be_nonnegative_and_lags_valid():
    ac = acf(_white_noise(50, 1), nlags=6)
    with pytest.raises(ValueError):
        ljung_box(ac.values, nobs=ac.nobs, model_df=-1, nlags=6)
    with pytest.raises(ValueError):
        ljung_box(ac.values, nobs=ac.nobs, nlags=0)
    with pytest.raises(ValueError):
        ljung_box(ac.values, nobs=ac.nobs, nlags=ac.nobs)
