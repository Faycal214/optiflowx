import numpy as np
import pytest

from stochx.timeseries import TimeSeries, Workfile, fit_ar, fit_ma, fit_arma


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


def test_eviews_starting_value_modes_are_supported():
    from stochx.timeseries.arma_estimation import make_starting_values
    y = np.arange(20.0)
    X = np.column_stack([np.ones(20), np.arange(20.0)])
    for mode in ["automatic", "eviews fixed", "random", "user-specified"]:
        user = [1.0, 0.5, 0.1, 0.2, 1.0] if mode == "user-specified" else None
        values = make_starting_values(y, X, (1,), (1,), method="ml", mode=mode, user=user, random_seed=1)
        assert values.shape == (5,)


def test_cls_starting_value_modes_and_backcasting_controls():
    y = _arma11(0.5, 0.3, 160, 48)
    wf = Workfile()
    wf.add("Y", y.values)
    for mode in ["OLS/TSLS", ".8 x OLS/TSLS", ".5 x OLS/TSLS", ".3 x OLS/TSLS", "Zero"]:
        result = wf.ls("Y C AR(1) MA(1)", arma_method="cls", arma_start=mode, backcast=True, name="CLS1")
        assert result.method.startswith("ARMA Conditional Least Squares")
        assert result.nobs > 0
    no_backcast = wf.ls("Y C AR(1) MA(1)", arma_method="cls", arma_start="Zero", backcast=False, name="CLS2")
    assert no_backcast.method.startswith("ARMA Conditional Least Squares")


def test_eviews_arma_structure_views_exist():
    y = _arma11(0.5, 0.3, 220, 49)
    wf = Workfile()
    wf.add("Y", y.values)
    result = wf.ls("Y C AR(1) MA(1)", name="ARMA1")
    roots = result.arma(type="root")
    acf = result.arma(type="acf", hrz=12)
    imp = result.arma(type="imp", hrz=12)
    freq = result.arma(type="freq", hrz=12)
    assert set(roots) == {"Inverted AR Roots", "Inverted MA Roots"}
    assert list(acf.columns) == ["Lag", "AC", "PAC"]
    assert list(imp.columns) == ["Period", "Impulse response"]
    assert list(freq.columns) == ["Frequency", "Spectrum"]
