import numpy as np

from stochx.timeseries import acf, ar, arma, ma, pacf, white_noise


def _reference_ac(values: np.ndarray, nlags: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    centered = values - mean
    denominator = float(np.dot(centered, centered))
    out = np.empty(nlags + 1, dtype=float)
    out[0] = 1.0
    for lag in range(1, nlags + 1):
        out[lag] = float(np.dot(centered[lag:], centered[:-lag]) / denominator)
    return out


def _reference_recursive_pacf(ac_values: np.ndarray) -> np.ndarray:
    ac_values = np.asarray(ac_values, dtype=float)
    nlags = len(ac_values) - 1
    out = np.ones(nlags + 1, dtype=float)
    previous = np.empty(0, dtype=float)
    for k in range(1, nlags + 1):
        numerator = float(ac_values[k])
        if k > 1:
            numerator -= float(np.dot(previous, ac_values[k - 1 : 0 : -1]))
        denominator = 1.0
        if k > 1:
            denominator -= float(np.dot(previous, ac_values[1:k]))
        phi_kk = numerator / denominator
        current = np.empty(k, dtype=float)
        if k == 1:
            current[0] = phi_kk
        else:
            current[:-1] = previous - phi_kk * previous[::-1]
            current[-1] = phi_kk
        previous = current
        out[k] = phi_kk
    return out


def _assert_ac_pacf_match(series, nlags=12):
    x = np.asarray(series.values, dtype=float)
    expected_ac = _reference_ac(x, nlags)
    expected_pac = _reference_recursive_pacf(expected_ac)

    ac_result = acf(series, nlags=nlags)
    pac_result = pacf(series, nlags=nlags)

    assert ac_result.nobs == len(x)
    assert pac_result.nobs == len(x)
    assert np.array_equal(ac_result.lags, np.arange(nlags + 1))
    assert np.array_equal(pac_result.lags, np.arange(nlags + 1))
    np.testing.assert_allclose(ac_result.values, expected_ac, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pac_result.values, expected_pac, rtol=1e-12, atol=1e-12)
    assert ac_result.values[0] == 1.0
    assert pac_result.values[0] == 1.0


def test_stage8_ac_pacf_white_noise():
    series = white_noise(512, rng=123)
    _assert_ac_pacf_match(series, nlags=16)
    assert np.max(np.abs(acf(series, nlags=16).values[1:])) < 0.2
    assert np.max(np.abs(pacf(series, nlags=16).values[2:])) < 0.2


def test_stage8_ac_pacf_ar1():
    series = ar(1, [0.7], 512, burnin=1000, rng=123)
    _assert_ac_pacf_match(series, nlags=16)
    result = pacf(series, nlags=8)
    assert abs(result.values[1]) > abs(result.values[2])
    assert abs(result.values[1] - 0.7) < 0.12


def test_stage8_ac_pacf_ma1():
    series = ma(1, [0.8], 512, burnin=200, rng=123)
    _assert_ac_pacf_match(series, nlags=16)
    result = acf(series, nlags=8)
    theoretical_rho1 = 0.8 / (1.0 + 0.8**2)
    assert abs(result.values[1] - theoretical_rho1) < 0.12


def test_stage8_ac_pacf_arma11():
    series = arma(1, 1, [0.6], [0.5], 512, burnin=1000, rng=123)
    _assert_ac_pacf_match(series, nlags=16)
    ac_result = acf(series, nlags=8)
    pac_result = pacf(series, nlags=8)
    assert np.isfinite(ac_result.values).all()
    assert np.isfinite(pac_result.values).all()
    assert abs(ac_result.values[1]) > abs(ac_result.values[4])


def test_stage8_public_ac_pacf_lag_zero_and_validation():
    series = white_noise(100, rng=7)
    ac_result = acf(series, nlags=0)
    pac_result = pacf(series, nlags=0)
    assert ac_result.lags.tolist() == [0]
    assert pac_result.lags.tolist() == [0]
    assert ac_result.values.tolist() == [1.0]
    assert pac_result.values.tolist() == [1.0]


def test_stage8_ac_uses_a_single_overall_mean():
    values = np.array([1.0, 2.0, 8.0, 4.0, 5.0, 6.0])
    result = acf(values, nlags=2)
    expected = _reference_ac(values, 2)
    np.testing.assert_allclose(result.values, expected, rtol=1e-12, atol=1e-12)
    separate_means_lag1 = np.corrcoef(values[1:], values[:-1])[0, 1]
    assert not np.isclose(result.values[1], separate_means_lag1)


def test_stage8_5_eviews_two_standard_error_bands_for_ordinary_series():
    series = white_noise(400, rng=21)
    for alpha in (0.01, 0.05, 0.10):
        ac_result = acf(series, nlags=8, alpha=alpha)
        pac_result = pacf(series, nlags=8, alpha=alpha)
        expected = 2.0 / np.sqrt(400)

        assert ac_result.band_method == "approx_two_standard_errors"
        assert pac_result.band_method == "approx_two_standard_errors"
        assert ac_result.band_multiplier == 2.0
        assert pac_result.band_multiplier == 2.0
        assert np.isclose(ac_result.band_standard_error, 1.0 / np.sqrt(400))
        assert np.isclose(pacf(series, nlags=8).band_standard_error, 1.0 / np.sqrt(400))
        np.testing.assert_allclose(ac_result.lower, -expected, rtol=1e-15, atol=1e-15)
        np.testing.assert_allclose(ac_result.upper, expected, rtol=1e-15, atol=1e-15)
        np.testing.assert_allclose(pac_result.lower, -expected, rtol=1e-15, atol=1e-15)
        np.testing.assert_allclose(pac_result.upper, expected, rtol=1e-15, atol=1e-15)


def test_stage8_5_residual_bands_use_shared_effective_nobs():
    residual = arma(p=1, q=1, phi=[0.45], theta=[0.25], n=300, rng=42)
    ac_result = acf(residual, nlags=8)
    pac_result = pacf(residual, nlags=8)
    expected = 2.0 / np.sqrt(300)

    assert ac_result.nobs == 300
    assert pac_result.nobs == 300
    assert ac_result.missing_count == 0
    assert pac_result.missing_count == 0
    np.testing.assert_allclose(ac_result.upper, expected, rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(pac_result.upper, expected, rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(ac_result.lower, -expected, rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(pac_result.lower, -expected, rtol=1e-15, atol=1e-15)
