import numpy as np

from stochx.timeseries import (
    DF_SPECIFICATIONS,
    SequentialDFResult,
    TimeSeries,
    adf,
    ar,
    correlogram,
    difference,
    dickey_fuller,
    dickey_fuller_sequential,
    estimate,
    fisher_seasonality_test,
    identify,
    ljung_box,
    moving_average,
    random_walk,
    seasonal_difference,
    white_noise,
)
from stochx.timeseries.forecasting import metrics
from stochx.timeseries.workfile import Workfile


def test_series_transformations_and_correlation():
    y = TimeSeries(np.arange(1.0, 101.0), name="Y", frequency="M")
    assert y.nobs == 100
    assert y.diff().nobs == 99
    assert y.log().nobs == 100
    assert y.lag(1).nmissing == 1
    assert y.acf(12).lags[-1] == 12
    assert y.pacf(12).lags[-1] == 12


def test_workfile_and_sample_workflow():
    y = TimeSeries(np.arange(1.0, 51.0), name="Y", frequency="M")
    dy = TimeSeries(np.r_[np.nan, np.diff(y.values)], name="D_Y", frequency="M")
    wf = Workfile()
    wf.add("Y", y.values)
    wf.add("D_Y", dy.values)
    wf.set_sample(5, 40)
    assert wf.nobs == 50
    assert wf.sample.start == 5
    assert wf.sample.stop == 41


def test_eviews_expression_and_generation_workflow():
    x = np.linspace(1.0, 50.0, 50)
    y = 2.0 + 0.5 * x
    wf = Workfile()
    wf.add("X", x)
    wf.add("Y", y)

    assert wf.eval("X").nobs == 50
    assert wf.eval("X(-1)").nmissing == 1
    assert np.isnan(wf.eval("X(-1)").values[0])
    assert wf.eval("X(1)").nmissing == 1
    assert wf.eval("D(X)").nobs == 49
    assert wf.eval("DLOG(X)").nobs == 49
    assert np.isclose(wf.eval("@mean(X)"), np.mean(x))
    assert np.isclose(wf.eval("@var(X)"), np.var(x, ddof=1))

    dx = wf.generate("DX", "D(X)")
    assert dx.nobs == 50
    assert dx.nmissing == 1
    assert np.allclose(dx.values[1:], np.diff(x))


def test_eviews_style_equation_and_unified_results():
    x = np.arange(1.0, 101.0)
    y = 3.0 + 2.0 * x + np.sin(x)
    wf = Workfile()
    wf.add("X", x)
    wf.add("Y", y)

    result = wf.ls("Y C X", name="EQ01")
    assert result.nobs == 100
    assert {"Coefficient", "Std. Error", "t-Statistic", "Prob."}.issubset(result.table().columns)
    assert "Equation: EQ01" in result.summary()
    interpretation = result.interpret()
    assert "statistically significant" in interpretation

    lagged = wf.ls("Y C X(-1)", name="EQ02")
    assert lagged.nobs == 99
    assert "X(-1)" in lagged.table().index


def test_simulation_and_identification_pipeline():
    y = ar(1, [0.6], 250, rng=0)
    ident = identify(y, nlags=12)
    assert ident["ACF"].nobs == 250
    result = estimate(y, p=1, d=0, q=0)
    assert result.params is not None
    assert hasattr(result, "table")
    assert hasattr(result, "interpret")
    corr = correlogram(y, nlags=12)
    assert {"AC", "PAC", "Q-Stat", "Prob."}.issubset(corr.columns)


def test_dickey_fuller_is_unaugmented():
    y = TimeSeries(random_walk(180, rng=3), name="Y")
    result = dickey_fuller(y, regression="c", alpha=0.05)
    assert result.test == "Dickey-Fuller Test"
    assert result.lags == 0
    assert result.regression == "c"
    assert result.critical_value == result.critical_values["5%"]


def test_adf_uses_regression_specific_nonstandard_critical_values():
    y = TimeSeries(random_walk(200, rng=11), name="Y")
    for regression in ("ct", "c", "n"):
        result = adf(y, regression=regression, lags=1, autolag=None, alpha=0.05)
        assert result.regression == regression
        assert result.specification_label == DF_SPECIFICATIONS[regression]["label"]
        assert {"1%", "5%", "10%"}.issubset(result.critical_values)
        assert result.critical_value == result.critical_values["5%"]
        assert result.decision in {"reject", "fail_to_reject"}
        expected = "reject" if result.statistic < result.critical_values["5%"] else "fail_to_reject"
        assert result.decision == expected
        assert "ordinary" not in result.decision_rule.lower()
        assert "Null hypothesis" in result.summary()
        assert "Decision rule" in result.summary()
        assert "not the decision rule" in result.summary()


def test_adf_summary_exposes_course_hypotheses_and_interpretation():
    y = TimeSeries(np.ones(120) + np.linspace(0.0, 1.0, 120), name="Y")
    result = adf(y, regression="c", lags=1, autolag=None)
    assert "γ = 0" in result.null_hypothesis
    assert "γ < 0" in result.alternative_hypothesis
    assert result.interpret() == result.conclusion
    table = result.table()
    assert {"Test Statistic", "Prob.*", "Critical Value", "Decision"}.issubset(table.columns)


def test_sequential_df_adf_runs_model_3_model_2_model_1_and_common_lag(monkeypatch):
    import stochx.timeseries.stationarity as stationarity

    stationary = TimeSeries(np.random.default_rng(7).normal(size=300), name="Y")
    calls = []
    original_adf = stationarity.adf

    def recording_adf(*args, **kwargs):
        calls.append((kwargs.get("regression"), kwargs.get("lags"), kwargs.get("autolag")))
        return original_adf(*args, **kwargs)

    monkeypatch.setattr(stationarity, "adf", recording_adf)
    result = stationarity.dickey_fuller_sequential(stationary, max_lags=2, autolag=None, alpha=0.05)
    assert isinstance(result, SequentialDFResult)

    regressions = [item.regression for item in result.tests]
    assert regressions in (["ct", "c"], ["ct", "c", "n"])

    visited_lags = [item.lags for item in result.tests]
    assert visited_lags == [2] * len(result.tests)
    assert [entry[0] for entry in calls] == regressions
    assert [entry[1] for entry in calls] == visited_lags
    assert [entry[2] for entry in calls] in (["None"] + [None] * (len(calls) - 1), [None] * len(calls))
    assert result.lag_order == 2
    assert len(result.table()) == len(result.tests)


def test_sequential_branch_model3_rejects_then_beta_retained(monkeypatch):
    import stochx.timeseries.stationarity as stationarity

    original_adf = stationarity.adf
    original_fit = stationarity._fit_df_regression

    def fake_adf(y, *, regression, lags=None, autolag=None, alpha=0.05):
        decision = "reject" if regression == "ct" else "fail_to_reject"
        from dataclasses import replace
        base = original_adf(np.arange(1.0, 80.0), regression=regression, lags=0, autolag=None, alpha=alpha)
        return replace(base, decision=decision, conclusion=f"forced {regression}")

    def fake_fit(x, regression, lags):
        base = original_fit(np.arange(1.0, 80.0), regression, 0)
        if regression == "ct":
            base["tvalues"] = np.array([0.0, 3.0, -5.0])
        return base

    monkeypatch.setattr(stationarity, "adf", fake_adf)
    monkeypatch.setattr(stationarity, "_fit_df_regression", fake_fit)
    result = stationarity.dickey_fuller_sequential(np.arange(1.0, 80.0), max_lags=0, autolag=None)
    assert result.selected.regression == "ct"
    assert "deterministic trend" in result.nature
    assert result.specification_tests[0].name == "Model 3 trend test"
    assert result.specification_tests[0].decision == "reject"


def test_sequential_branch_model3_rejects_beta_insignificant_then_model2(monkeypatch):
    import stochx.timeseries.stationarity as stationarity
    from dataclasses import replace

    original_adf = stationarity.adf
    original_fit = stationarity._fit_df_regression

    def fake_adf(y, *, regression, lags=None, autolag=None, alpha=0.05):
        base = original_adf(np.arange(1.0, 80.0), regression=regression, lags=0, autolag=None, alpha=alpha)
        return replace(base, decision="reject" if regression in {"ct", "c"} else "fail_to_reject")

    def fake_fit(x, regression, lags):
        base = original_fit(np.arange(1.0, 80.0), regression, 0)
        if regression == "ct":
            base["tvalues"] = np.array([0.0, 0.2, -5.0])
        elif regression == "c":
            base["tvalues"] = np.array([3.0, -5.0])
        return base

    monkeypatch.setattr(stationarity, "adf", fake_adf)
    monkeypatch.setattr(stationarity, "_fit_df_regression", fake_fit)
    result = stationarity.dickey_fuller_sequential(np.arange(1.0, 80.0), max_lags=0, autolag=None)
    assert result.selected.regression == "c"
    assert [item.name for item in result.specification_tests[:2]] == ["Model 3 trend test", "Model 2 constant test"]
    assert result.specification_tests[0].decision == "fail_to_reject"
    assert result.specification_tests[1].decision == "reject"


def test_sequential_branch_model3_unit_root_then_f3(monkeypatch):
    import stochx.timeseries.stationarity as stationarity
    from dataclasses import replace

    original_adf = stationarity.adf

    def fake_adf(y, *, regression, lags=None, autolag=None, alpha=0.05):
        base = original_adf(np.arange(1.0, 80.0), regression=regression, lags=0, autolag=None, alpha=alpha)
        return replace(base, decision="fail_to_reject")

    def fake_f(x, regression, lags, alpha):
        return stationarity.SpecificationTestResult(
            name="Model 3 joint F test" if regression == "ct" else "Model 2 joint F test",
            null_hypothesis="joint H0", alternative_hypothesis="joint H1",
            statistic=99.0, critical_value=6.0 if regression == "ct" else 4.0,
            decision="reject", alpha=alpha, source="test",
        )

    monkeypatch.setattr(stationarity, "adf", fake_adf)
    monkeypatch.setattr(stationarity, "_joint_f_test", fake_f)
    result = stationarity.dickey_fuller_sequential(np.arange(1.0, 80.0), max_lags=0, autolag=None)
    assert result.selected.regression == "ct"
    assert result.specification_tests[0].name == "Model 3 joint F test"
    assert result.specification_tests[0].decision == "reject"
    assert result.nature.startswith("I(1)")
    assert "Model 3" in result.nature


def test_sequential_branch_model3_f3_accepts_then_model2_f2(monkeypatch):
    import stochx.timeseries.stationarity as stationarity
    from dataclasses import replace

    original_adf = stationarity.adf

    def fake_adf(y, *, regression, lags=None, autolag=None, alpha=0.05):
        base = original_adf(np.arange(1.0, 80.0), regression=regression, lags=0, autolag=None, alpha=alpha)
        return replace(base, decision="fail_to_reject")

    def fake_f(x, regression, lags, alpha):
        return stationarity.SpecificationTestResult(
            name="Model 3 joint F test" if regression == "ct" else "Model 2 joint F test",
            null_hypothesis="joint H0", alternative_hypothesis="joint H1",
            statistic=1.0, critical_value=6.0 if regression == "ct" else 4.0,
            decision="fail_to_reject" if regression == "ct" else "reject", alpha=alpha, source="test",
        )

    monkeypatch.setattr(stationarity, "adf", fake_adf)
    monkeypatch.setattr(stationarity, "_joint_f_test", fake_f)
    result = stationarity.dickey_fuller_sequential(np.arange(1.0, 80.0), max_lags=0, autolag=None)
    assert [item.name for item in result.specification_tests] == ["Model 3 joint F test", "Model 2 joint F test"]
    assert result.selected.regression == "c"
    assert result.nature.endswith("DS candidate)")


def test_sequential_interpretation_is_course_faithful(monkeypatch):
    import stochx.timeseries.stationarity as stationarity
    from dataclasses import replace

    original_adf = stationarity.adf

    def fake_adf(y, *, regression, lags=None, autolag=None, alpha=0.05):
        base = original_adf(np.arange(1.0, 80.0), regression=regression, lags=0, autolag=None, alpha=alpha)
        return replace(base, decision="fail_to_reject")

    monkeypatch.setattr(stationarity, "adf", fake_adf)
    monkeypatch.setattr(
        stationarity,
        "_joint_f_test",
        lambda x, regression, lags, alpha: stationarity.SpecificationTestResult(
            name="Model 3 joint F test" if regression == "ct" else "Model 2 joint F test",
            null_hypothesis="joint H0", alternative_hypothesis="joint H1",
            statistic=1.0, critical_value=6.0 if regression == "ct" else 4.0,
            decision="fail_to_reject", alpha=alpha, source="test",
        ),
    )
    result = stationarity.dickey_fuller_sequential(np.arange(1.0, 80.0), max_lags=0, autolag=None)
    text = result.interpret()
    assert "Model 1" in text or result.selected.regression == "n"
    assert "unit-root null" in text


def test_stationarity_and_forecast_metrics():
    y = random_walk(120, rng=0)
    result = adf(y, regression="c", lags=1, autolag=None)
    assert np.isfinite(result.statistic)
    assert result.pvalue is not None
    assert {"1%", "5%", "10%"}.issubset(result.critical_values)
    assert "Augmented Dickey-Fuller Test" in result.summary()
    assert difference(y, 1).nobs == 119
    ma = moving_average(y, 5)
    assert ma.nobs == 120
    err = metrics([1, 2, 3], [1, 1, 4])
    assert err.rmse > 0


def test_seasonality_and_residual_test():
    seasonal = TimeSeries(np.tile([10.0, 12.0, 15.0, 11.0], 20), name="S", frequency="Q")
    seasonal_d = seasonal_difference(seasonal, 4)
    assert seasonal_d.nobs == 76
    fisher = fisher_seasonality_test(seasonal, 4)
    assert fisher["reject_seasonality_null"]
    lb = ljung_box(white_noise(200, rng=0), lags=12)
    assert np.isfinite(lb.statistic)
