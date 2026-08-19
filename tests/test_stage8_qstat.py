import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from stochx.timeseries import ar, arma, correlogram, white_noise
from stochx.timeseries.correlogram import ljung_box
from stochx.timeseries.correlation import acf


def _reference_q(ac_values, nobs, nlags):
    q = []
    cumulative = 0.0
    for lag in range(1, nlags + 1):
        cumulative += ac_values[lag] ** 2 / (nobs - lag)
        q.append(nobs * (nobs + 2) * cumulative)
    return np.asarray(q)


def test_ljung_box_matches_independent_formula_for_white_noise():
    y = white_noise(250, rng=123)
    ac = acf(y, nlags=12)
    result = ljung_box(ac.values, nobs=ac.nobs, nlags=12)
    expected_q = _reference_q(ac.values, ac.nobs, 12)
    expected_p = np.asarray([chi2.sf(expected_q[k - 1], k) for k in range(1, 13)])

    assert np.allclose(result.q_stat, expected_q, rtol=1e-12, atol=1e-12)
    assert np.allclose(result.pvalues, expected_p, rtol=1e-12, atol=1e-12)
    assert np.array_equal(result.df, np.arange(1, 13))


def test_correlogram_exposes_qstat_and_probability_columns():
    y = white_noise(150, rng=7)
    result = correlogram(y, nlags=10)

    assert isinstance(result, pd.DataFrame)
    assert list(result["Lag"]) == list(range(1, 11))
    assert {"AC", "PAC", "Q-Stat", "Prob.", "DF"}.issubset(result.columns)
    assert np.isfinite(result["Q-Stat"]).all()
    assert np.isfinite(result["Prob."]).all()


def test_residual_model_df_adjusts_degrees_of_freedom_and_probabilities():
    y = arma(p=1, q=1, phi=[0.45], theta=[0.25], n=300, rng=42)
    ac = acf(y, nlags=8)
    result = ljung_box(ac.values, nobs=ac.nobs, model_df=2, nlags=8)

    assert np.array_equal(result.df, np.arange(1, 9) - 2)
    assert np.isnan(result.pvalues[:2]).all()
    assert np.isfinite(result.pvalues[2:]).all()

    expected_q = _reference_q(ac.values, ac.nobs, 8)
    assert np.allclose(result.q_stat, expected_q, rtol=1e-12, atol=1e-12)
    for i, df in enumerate(result.df):
        if df > 0:
            assert np.isclose(result.pvalues[i], chi2.sf(expected_q[i], int(df)), rtol=1e-12, atol=1e-12)


def test_model_df_must_be_nonnegative_and_lags_valid():
    ac = acf(white_noise(50, rng=1), nlags=6)
    with pytest.raises(ValueError):
        ljung_box(ac.values, nobs=ac.nobs, model_df=-1, nlags=6)
    with pytest.raises(ValueError):
        ljung_box(ac.values, nobs=ac.nobs, nlags=0)
    with pytest.raises(ValueError):
        ljung_box(ac.values, nobs=ac.nobs, nlags=ac.nobs)


def test_ljung_box_q_statistics_are_cumulative():
    y = ar(p=1, phi=[0.65], n=220, rng=19)
    ac = acf(y, nlags=12)
    result = ljung_box(ac.values, nobs=ac.nobs, nlags=12)
    assert np.all(np.diff(result.q_stat) >= -1e-12)
