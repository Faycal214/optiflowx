import numpy as np
import pytest

from stochx.timeseries import CorrelogramResult, arma, correlogram, interpret_correlogram, white_noise


def _result_with_spikes() -> CorrelogramResult:
    lags = np.arange(1, 5)
    ac = np.array([0.30, 0.01, -0.25, 0.00])
    pac = np.array([0.01, 0.28, 0.00, -0.24])
    q = np.array([2.0, 3.0, 5.0, 6.0])
    p = np.array([0.20, 0.08, 0.04, 0.02])
    df = lags.copy()
    lower = np.full(4, -0.20)
    upper = np.full(4, 0.20)
    return CorrelogramResult(
        lags=lags,
        ac=ac,
        pac=pac,
        q_stat=q,
        pvalues=p,
        df=df,
        nobs=100,
        nlags=4,
        model_df=0,
        alpha=0.05,
        series_name="Y",
        ac_lower=lower,
        ac_upper=upper,
        pac_lower=lower,
        pac_upper=upper,
    )


def test_stage8_8_ordinary_interpretation_lists_significant_ac_pac_spikes_and_qtest():
    result = _result_with_spikes()
    text = interpret_correlogram(result)

    assert "Ordinary correlogram for Y (100 observations)." in text
    assert "Significant AC spikes at lag(s) 1, 3." in text
    assert "Significant PAC spikes at lag(s) 2, 4." in text
    assert "Ljung-Box Q=6.0000 with df=4 and p=0.02" in text
    assert "rejects the no-autocorrelation null at the 5% level" in text


def test_stage8_8_residual_interpretation_reports_model_df_and_ignores_nonpositive_probabilities():
    y = arma(p=1, q=1, phi=[0.45], theta=[0.25], n=160, rng=7)
    result = correlogram(y, nlags=6, model_df=2)
    text = interpret_correlogram(result)

    assert "Residual correlogram" in text
    assert "The Ljung-Box degrees of freedom are adjusted by model_df=2." in text
    assert "df=4" in text
    assert "p=" in text


def test_stage8_8_no_significant_spikes_is_reported_cleanly():
    result = correlogram(white_noise(120, rng=21), nlags=6)
    text = interpret_correlogram(result)

    assert "No AC spikes are outside the displayed confidence bands." in text or "Significant AC spikes" in text
    assert "No PAC spikes are outside the displayed confidence bands." in text or "Significant PAC spikes" in text
    assert "Ljung-Box" in text


def test_stage8_8_max_spikes_and_alpha_are_deterministic():
    result = _result_with_spikes()
    text = interpret_correlogram(result, alpha=0.10, max_spikes=1)

    assert "Significant AC spikes at lag(s) 1." in text
    assert "Significant PAC spikes at lag(s) 2." in text
    assert "does not reject" in text


def test_stage8_8_interpretation_does_not_mutate_frozen_result_or_table():
    result = correlogram(white_noise(90, rng=4), nlags=5)
    before = result.table().copy()
    _ = interpret_correlogram(result)

    np.testing.assert_array_equal(result.AC, result.ac)
    np.testing.assert_array_equal(result.Q_Stat, result.q_stat)
    assert result.table().equals(before)


def test_stage8_8_validates_inputs():
    result = correlogram(white_noise(50, rng=1), nlags=4)
    with pytest.raises(TypeError):
        interpret_correlogram(object())
    with pytest.raises(ValueError):
        interpret_correlogram(result, alpha=1.0)
    with pytest.raises(ValueError):
        interpret_correlogram(result, max_spikes=0)
