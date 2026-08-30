import numpy as np

from stochx.timeseries import TimeSeries, Workfile, fit_arima


def test_eviews_forecast_result_has_forecast_error_standard_errors():
    y = TimeSeries(np.cumsum(np.random.default_rng(10001).normal(size=140)), name="Y")
    result = fit_arima(y, 1, 1, 0)
    frame = result.forecast(8)
    assert {"Forecast", "Std. Error", "Lower", "Upper"} <= set(frame.columns)
    assert np.all(frame["Upper"].to_numpy() >= frame["Forecast"].to_numpy())
    assert np.all(frame["Lower"].to_numpy() <= frame["Forecast"].to_numpy())


def test_equation_static_fit_and_dynamic_forecast_are_distinct_operations():
    rng = np.random.default_rng(10002)
    y = np.zeros(180)
    x = rng.normal(size=180)
    eps = rng.normal(size=180)
    for t in range(1, 180):
        y[t] = 0.5 + 0.35 * x[t] + 0.55 * y[t - 1] + eps[t]

    wf = Workfile()
    wf.add("Y", y)
    wf.add("X", x)
    eq = wf.ls("Y C X Y(-1)", name="EQF")

    static = eq.fit(start=150, end=179)
    dynamic = eq.forecast(start=150, end=179, dynamic=True)

    assert len(static) == 30
    assert len(dynamic) == 30
    assert static.attrs["dynamic"] is False
    assert dynamic.attrs["dynamic"] is True


def test_structural_forecast_ignores_arma_terms():
    rng = np.random.default_rng(10003)
    x = rng.normal(size=180)
    eps = rng.normal(size=180)
    y = 1.2 + 0.8 * x + eps

    wf = Workfile()
    wf.add("Y", y)
    wf.add("X", x)
    eq = wf.ls("Y C X AR(1)", name="EQSF")

    structural = eq.forecast(6, structural=True)
    dynamic = eq.forecast(6, structural=False)

    assert structural.attrs["structural"] is True
    assert dynamic.attrs["structural"] is False


def test_forecast_evaluation_matches_eviews_measures_and_decomposition():
    wf = Workfile()
    wf.add("Y", np.arange(12.0))
    eq = wf.ls("Y C", name="EQEV")

    metrics = eq.forecast_evaluation(
        forecast=[0, 1, 2, 3, 4],
        actual=[0, 2, 2, 4, 5],
    )

    assert {"RMSE", "MAE", "MAPE", "Theil Inequality Coefficient",
            "Bias Proportion", "Variance Proportion",
            "Covariance Proportion"} <= set(metrics)
    assert metrics["RMSE"] > 0
