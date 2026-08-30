import numpy as np

from stochx.timeseries import (
    CorrelogramResult,
    correlogram,
    fit_ar,
    residual_correlogram,
    residual_diagnostics_correlogram,
)


def test_stage8_12_direct_residual_correlogram_reuses_frozen_result_contract():
    rng = np.random.default_rng(7)
    eps = rng.normal(size=160)
    values = np.empty(160)
    values[0] = eps[0]
    for t in range(1, 160):
        values[t] = 0.45 * values[t - 1] + eps[t] + 0.25 * eps[t - 1]
    residuals = values
    result = residual_correlogram(residuals, lags=6, model_df=2, alpha=0.05)
    reference = correlogram(residuals, nlags=6, model_df=2, alpha=0.05)

    assert isinstance(result, CorrelogramResult)
    assert result.nobs == reference.nobs == 160
    assert result.nlags == result.lags.size == 6
    assert result.model_df == 2
    np.testing.assert_array_equal(result.DF, reference.DF)
    np.testing.assert_allclose(result.AC, reference.AC, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.PAC, reference.PAC, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.Q_Stat, reference.Q_Stat, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.Prob, reference.Prob, rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(result.ac_lower, reference.ac_lower, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.ac_upper, reference.ac_upper, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.pac_lower, reference.pac_lower, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.pac_upper, reference.pac_upper, rtol=1e-12, atol=1e-12)
    assert not result.ac.flags.writeable
    assert not result.q_stat.flags.writeable


def test_stage8_12_model_diagnostics_residual_correlogram_uses_p_plus_q_model_df():
    y = np.sin(np.linspace(0.0, 18.0, 180)) + 0.15 * np.cos(np.linspace(0.0, 31.0, 180))
    fitted = __import__("stochx.timeseries", fromlist=["fit_arma"]).fit_arma(y, 1, 1)

    result = fitted.residual_correlogram(lags=8, alpha=0.05)
    reference = correlogram(fitted.residuals, nlags=8, model_df=2, alpha=0.05)

    assert isinstance(result, CorrelogramResult)
    assert result.model_df == 2
    np.testing.assert_array_equal(result.DF, np.arange(1, 9) - 2)
    np.testing.assert_allclose(result.AC, reference.AC, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.PAC, reference.PAC, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.Q_Stat, reference.Q_Stat, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.Prob, reference.Prob, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_stage8_12_model_diagnostics_correlogram_alias_matches_residual_correlogram():
    y = np.sin(np.linspace(0.0, 14.0, 140)) + 0.1 * np.cos(np.linspace(0.0, 9.0, 140))
    fitted = fit_ar(y, 1)

    residual_result = fitted.residual_correlogram(lags=6)
    alias_result = fitted.correlogram(lags=6)

    np.testing.assert_array_equal(alias_result.lags, residual_result.lags)
    np.testing.assert_allclose(alias_result.AC, residual_result.AC, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(alias_result.PAC, residual_result.PAC, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(alias_result.Q_Stat, residual_result.Q_Stat, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(alias_result.Prob, residual_result.Prob, rtol=1e-12, atol=1e-12, equal_nan=True)
    assert alias_result.model_df == residual_result.model_df == 1


def test_stage8_12_diagnostics_helper_matches_tsresult_method():
    y = np.sin(np.linspace(0.0, 12.0, 120))
    fitted = fit_ar(y, 1)

    direct = residual_diagnostics_correlogram(fitted, lags=5, alpha=0.10)
    method = fitted.residual_correlogram(lags=5, alpha=0.10)

    assert direct.alpha == method.alpha == 0.10
    assert direct.model_df == method.model_df == 1
    np.testing.assert_array_equal(direct.DF, method.DF)
    np.testing.assert_allclose(direct.AC, method.AC, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(direct.PAC, method.PAC, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(direct.Q_Stat, method.Q_Stat, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(direct.Prob, method.Prob, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_stage8_12_residual_integration_preserves_numeric_table_and_aliases():
    y = np.sin(np.linspace(0.0, 16.0, 150)) + 0.05 * np.cos(np.linspace(0.0, 21.0, 150))
    fitted = fit_ar(y, 1)
    result = fitted.residual_correlogram(lags=6)
    before = result.table().copy(deep=True)

    _ = result.table()
    _ = result.summary()
    _ = result.interpret()

    assert result.table().equals(before)
    assert result.AC is result.ac
    assert result.PAC is result.pac
    assert result.Q_Stat is result.q_stat
    assert result.QStat is result.q_stat
    assert result.Prob is result.pvalues
    assert result.PValues is result.pvalues
    assert result.DF is result.df
    assert not result.ac.flags.writeable
    assert not result.pvalues.flags.writeable
