import numpy as np

from stochx.timeseries import Workfile


def test_eviews_equation_object_keeps_mean_coefficients_before_arma_terms():
    rng = np.random.default_rng(70)
    y = np.zeros(220)
    x = rng.normal(size=220)
    e = rng.normal(size=220)
    for t in range(1, 220):
        e[t] += 0.55 * e[t - 1]
        y[t] = 1.5 + 0.8 * x[t] + e[t]
    wf = Workfile()
    wf.add("Y", y)
    wf.add("X", x)
    eq = wf.ls("Y C X AR(1)", name="EQ07")
    assert list(eq.params.index)[:2] == ["C", "X"]
    assert "AR(1)" in eq.params.index
    assert "SIGMASQ" in eq.params.index
    assert eq.covariance_method == "outer product of gradients (OPG)"


def test_eviews_equation_object_exposes_residuals_and_diagnostics():
    rng = np.random.default_rng(71)
    wf = Workfile()
    wf.add("Y", 2.0 + rng.normal(size=180))
    eq = wf.ls("Y C", name="EQ08")
    assert len(eq.residuals) == eq.nobs
    assert len(eq.fittedvalues) == eq.nobs
    corr = eq.residual_correlogram(lags=8)
    assert corr.nobs > 0
    assert "Q-Stat" in corr.table().columns


def test_eviews_equation_forecast_supports_structural_switch():
    rng = np.random.default_rng(72)
    x = rng.normal(size=160)
    e = rng.normal(size=160)
    y = 2.0 + 0.7 * x + e
    wf = Workfile()
    wf.add("Y", y)
    wf.add("X", x)
    eq = wf.ls("Y C X AR(1)", name="EQ09")
    structural = eq.forecast(8, structural=True)
    assert len(structural) == 8
    assert "Forecast" in structural.columns


def test_eviews_equation_forecast_evaluation_returns_standard_measures():
    wf = Workfile()
    wf.add("Y", np.arange(20.0))
    eq = wf.ls("Y C", name="EQ10")
    result = eq.forecast(5, structural=True)
    metrics = eq.forecast_evaluation(result["Forecast"], actual=np.arange(20.0, 25.0))
    assert {"RMSE", "MAE", "MAPE", "Mean Error", "Theil U"} <= set(metrics)