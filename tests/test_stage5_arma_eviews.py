import numpy as np
import pytest

from stochx.timeseries import TimeSeries, fit_ar, fit_ma, fit_arma


def _arma11(phi: float, theta: float, n: int, seed: int) -> TimeSeries:
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=n)
    y = np.empty(n)
    y[0] = eps[0]
    for t in range(1, n):
        y[t] = phi * y[t - 1] + eps[t] + theta * eps[t - 1]
    return TimeSeries(y, name="Y")


def test_ar_defaults_to_maximum_likelihood_and_reports_eviews_method():
    y = _arma11(0.6, 0.0, 250, 42)
    result = fit_ar(y, 1)
    assert result.method == "Maximum Likelihood"
    assert result.order == (1, 0, 0)
    assert "ar.L1" in result.result.param_names
    assert np.isfinite(result.params.iloc[0])


def test_ma_and_arma_use_ml_by_default():
    y = _arma11(0.0, 0.5, 250, 43)
    ma = fit_ma(y, 1)
    arma = fit_arma(y, 1, 1)
    assert ma.method == "Maximum Likelihood"
    assert arma.method == "Maximum Likelihood"
    assert ma.order == (0, 0, 1)
    assert arma.order == (1, 0, 1)


def test_eviews_parameter_labels_are_exposed_without_losing_backend_names():
    y = _arma11(0.5, 0.4, 220, 44)
    result = fit_arma(y, 1, 1)
    labels = set(result.params_eviews.index)
    assert "AR(1)" in labels
    assert "MA(1)" in labels
    assert "SIGMASQ" in labels


def test_roots_are_reported_in_eviews_inverted_form():
    y = _arma11(0.5, 0.4, 300, 45)
    result = fit_arma(y, 1, 1)
    backend_ar = np.asarray(result.result.arroots, dtype=complex)
    backend_ma = np.asarray(result.result.maroots, dtype=complex)
    reported = result.roots()
    np.testing.assert_allclose(reported["AR roots"], 1.0 / backend_ar)
    np.testing.assert_allclose(reported["MA roots"], 1.0 / backend_ma)


def test_non_ml_methods_are_not_silently_mapped_to_ml():
    y = _arma11(0.4, 0.2, 120, 46)
    with pytest.raises(ValueError):
        fit_ar(y, 1, method="yule_walker")
    with pytest.raises(ValueError):
        fit_arma(y, 1, 1, method="cls")

def test_direct_arma_api_preserves_sparse_lags():
    y = _arma11(0.4, 0.2, 180, 47)
    result = fit_arma(y, [1, 4], [2, 5])
    names = set(result.params.index)
    assert "AR(1)" in names
    assert "AR(4)" in names
    assert "MA(2)" in names
    assert "MA(5)" in names
    assert "AR(2)" not in names
    assert "MA(1)" not in names
