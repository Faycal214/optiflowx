import numpy as np

from stochx.timeseries import TimeSeries, Workfile, fit_arima, fit_sarima


def _seasonal_series(n=160, s=12, seed=123):
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    eps = rng.normal(size=n)
    for t in range(n):
        y[t] = 0.6 * y[t - 1] + 0.35 * np.sin(2 * np.pi * t / s) + eps[t] if t else eps[t]
    return y


def test_arima_defaults_to_eviews_ml_bfgs_opg():
    y = TimeSeries(np.cumsum(np.random.default_rng(1).normal(size=140)), name="Y")
    result = fit_arima(y, 1, 1, 0)
    assert result.method == "Maximum Likelihood"
    assert result.order == (1, 1, 0)
    assert np.isfinite(result.statistics()["Log likelihood"])


def test_sarima_uses_multiplicative_seasonal_structure():
    y = TimeSeries(_seasonal_series(), name="Y")
    result = fit_sarima(y, (1, 0, 1), (1, 0, 1, 12))
    assert result.seasonal_order == (1, 0, 1, 12)
    labels = set(result.params.index)
    assert "AR(1)" in labels
    assert "MA(1)" in labels
    assert "SAR(12)" in labels
    assert "SMA(12)" in labels
    assert "SIGMASQ" in labels


def test_workfile_sarima_infers_periodicity_from_frequency():
    y = _seasonal_series()
    wf = Workfile(frequency="M")
    wf.add("Y", y)
    result = wf.sarima("Y", (1, 1, 0), (1, 1, 0, 12))
    assert result.seasonal_order == (1, 1, 0, 12)
    assert result.method == "Maximum Likelihood"


def test_arima_forecast_supports_dynamic_and_static_paths():
    y = TimeSeries(np.cumsum(np.random.default_rng(2).normal(size=120)), name="Y")
    result = fit_arima(y, 1, 1, 0)
    static = result.forecast(6, dynamic=False)
    dynamic = result.forecast(6, dynamic=True)
    assert len(static) == 6
    assert len(dynamic) == 6
    assert "Forecast" in static.columns


def test_eviews_seasonal_terms_are_parsed_and_estimated_in_equations():
    wf = Workfile(frequency="M")
    wf.add("Y", _seasonal_series(n=180))
    regressors, process = __import__("stochx.timeseries.arma_errors", fromlist=["parse_error_terms"]).parse_error_terms(
        ["C", "SAR(1)", "SMA(1)"]
    )
    assert regressors == ["C"]
    assert process.sar == (1,)
    assert process.sma == (1,)
    result = wf.ls("Y C SAR(1) SMA(1)", name="SARIMA1")
    labels = set(result.params.index)
    assert "SAR(1)" in labels
    assert "SMA(1)" in labels
