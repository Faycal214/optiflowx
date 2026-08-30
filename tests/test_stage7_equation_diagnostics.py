import numpy as np
import pytest

from stochx.timeseries import Workfile


def test_eviews_equation_residual_diagnostics_family():
    rng = np.random.default_rng(901)
    x = rng.normal(size=220)
    e = rng.normal(size=220)
    y = 1.5 + 0.8 * x + e

    wf = Workfile()
    wf.add("Y", y)
    wf.add("X", x)
    eq = wf.ls("Y C X", name="EQD")

    report = eq.diagnostics(lags=8)

    assert "Correlogram-Q statistics" in report
    assert "Squared residual correlogram" in report
    assert "Histogram-Normality" in report
    assert "Heteroskedasticity" in report
    assert "Serial Correlation LM" in report
    assert "Q-Stat" in report["Correlogram-Q statistics"].table().columns


def test_eviews_squared_residual_correlogram_is_separate_view():
    rng = np.random.default_rng(902)
    wf = Workfile()
    wf.add("Y", rng.normal(size=180))
    eq = wf.ls("Y C", name="EQSQ")

    result = eq.squared_residual_correlogram(lags=6)
    assert result.lags[-1] == 6


def test_eviews_heteroskedasticity_test_family():
    rng = np.random.default_rng(903)
    x = rng.normal(size=220)
    y = 2 + 0.5 * x + rng.normal(size=220)

    wf = Workfile()
    wf.add("Y", y)
    wf.add("X", x)
    eq = wf.ls("Y C X", name="EQH")

    for test in ["BPG", "Harvey", "Glejser", "ARCH", "White"]:
        result = eq.heteroskedasticity(test=test, lags=4)
        assert "p-value" in result
        assert "LM statistic" in result


def test_eviews_serial_correlation_lm_reports_obs_r_squared_and_f():
    rng = np.random.default_rng(904)
    x = rng.normal(size=220)
    y = 1 + 0.4 * x + rng.normal(size=220)

    wf = Workfile()
    wf.add("Y", y)
    wf.add("X", x)
    eq = wf.ls("Y C X", name="EQLM")

    result = eq.serial_correlation_lm(lags=4)
    assert "Obs*R-squared" in result
    assert "F-statistic" in result
    assert result["df"] == 4


def test_eviews_stability_diagnostics_are_ols_only():
    rng = np.random.default_rng(905)
    x = rng.normal(size=180)
    y = 1 + 0.3 * x + rng.normal(size=180)
    wf = Workfile()
    wf.add("Y", y)
    wf.add("X", x)
    eq = wf.ls("Y C X", name="EQST")
    result = eq.stability_diagnostics(breakpoint=90, forecast_start=120)
    assert "recursive" in result
    assert "CUSUM" in result["recursive"]
    assert "CUSUMSQ" in result["recursive"]
    assert "Chow breakpoint" in result
    assert "Chow forecast" in result

    wf2 = Workfile()
    wf2.add("Y", y)
    arma_eq = wf2.ls("Y C AR(1)", name="EQAR")
    with pytest.raises(ValueError):
        arma_eq.stability_diagnostics()
