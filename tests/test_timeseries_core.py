import scipy
import pytest
import pandas as pd

import numpy as np

from stochx.timeseries import (
    DF_SPECIFICATIONS,
    SequentialDFResult,
    TimeSeries,
    adf,
    correlogram,
    difference,
    dickey_fuller,
    dickey_fuller_sequential,
    estimate,
    fisher_seasonality_test,
    identify,
    ljung_box,
    moving_average,
    seasonal_difference,
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


def test_identification_and_estimation_pipeline():
    rng = np.random.default_rng(0)
    y = np.empty(250)
    eps = rng.normal(size=250)
    y[0] = eps[0]
    for t in range(1, 250):
        y[t] = 0.6 * y[t - 1] + eps[t]
    y = TimeSeries(y, name="Y")
    ident = identify(y, nlags=12)
    assert ident["ACF"].nobs == 250
    result = estimate(y, p=1, d=0, q=0)
    assert result.params is not None
    assert hasattr(result, "table")
    assert hasattr(result, "interpret")
    corr = correlogram(y, nlags=12)
    assert {"AC", "PAC", "Q-Stat", "Prob."}.issubset(corr.columns)


def test_dickey_fuller_is_unaugmented():
    rng = np.random.default_rng(3)
    eps = rng.normal(size=180)
    values = np.cumsum(eps)
    y = TimeSeries(values, name="Y")
    result = dickey_fuller(y, regression="c", alpha=0.05)
    assert result.test == "Dickey-Fuller Test"
    assert result.lags == 0
    assert result.regression == "c"
    assert result.critical_value == result.critical_values["5%"]


def test_adf_uses_eviews_critical_values_for_fixed_lag():
    rng = np.random.default_rng(11)
    values = np.cumsum(rng.normal(size=200))
    y = TimeSeries(values, name="Y")
    for regression in ("ct", "c", "n"):
        result = adf(y, regression=regression, lags=1, autolag=None, alpha=0.05)
        assert result.regression == regression
        assert result.specification_label == DF_SPECIFICATIONS[regression]["label"]
        assert {"1%", "5%", "10%"}.issubset(result.critical_values)
        assert result.critical_value == result.critical_values["5%"]
        assert result.decision in {"reject", "fail_to_reject"}
        expected = "reject" if result.statistic < result.critical_values["5%"] else "fail_to_reject"
        assert result.decision == expected
        assert result.lag_selection_method == "Fixed"
        assert "MacKinnon" in result.critical_value_source


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
    original_adf = stationarity._course_adf

    def recording_adf(*args, **kwargs):
        calls.append((kwargs.get("regression"), kwargs.get("lags"), kwargs.get("autolag")))
        return original_adf(*args, **kwargs)

    monkeypatch.setattr(stationarity, "_course_adf", recording_adf)
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

    monkeypatch.setattr(stationarity, "_course_adf", fake_adf)
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
    rng = np.random.default_rng(0)
    y = TimeSeries(np.cumsum(rng.normal(size=120)), name="Y")
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
    lb = ljung_box(TimeSeries(np.random.default_rng(0).normal(size=200), name="WN"), lags=12)
    assert np.isfinite(lb.statistic)


def test_workfile_infers_and_validates_shared_index():
    index = list(pd.date_range("2020-01-01", periods=12, freq="MS"))
    wf = Workfile(frequency="M")
    wf.add("X", np.arange(12.0), index=index)
    wf.add("Y", np.arange(12.0) + 1.0)
    assert wf["Y"].index == wf["X"].index
    with pytest.raises(ValueError, match="same index"):
        wf.add("Z", np.arange(12.0), index=list(pd.date_range("2020-02-01", periods=12, freq="MS")))


def test_workfile_accepts_eviews_period_sample_labels():
    index = list(pd.date_range("2010-01-01", periods=24, freq="MS"))
    wf = Workfile.from_dataframe(
        pd.DataFrame({"X": np.arange(24.0)}),
        index=index,
        frequency="M",
    )
    wf.set_sample("2011M1 2011M12")
    assert wf.sample == slice(12, 24)


def test_eviews_trend_expression_starts_at_zero():
    wf = Workfile()
    wf.add("X", np.arange(5.0))
    trend = wf.eval("@TREND")
    assert np.array_equal(trend.values, np.arange(5.0))
    generated = wf.generate("T", "@TREND")
    assert np.array_equal(generated.values, np.arange(5.0))


def test_workfile_series_names_are_case_insensitive_like_eviews():
    wf = Workfile()
    wf.add("GDP", np.arange(5.0))
    assert "gdp" in wf
    assert np.array_equal(wf["gDp"].values, np.arange(5.0))
    with pytest.raises(ValueError, match="reserved"):
        wf.add("c", np.ones(5))


def test_workfile_smpl_supports_eviews_if_conditions():
    wf = Workfile()
    wf.add("X", np.arange(-2.0, 5.0))
    wf.smpl("smpl @all if X>=0")
    assert wf.sample_mask is not None
    assert wf.sample_series("X").nobs == 5
    assert np.array_equal(wf.sample_series("X").values, np.arange(5.0))


def test_expression_supports_eviews_comparisons_and_logic():
    wf = Workfile()
    wf.add("X", np.arange(5.0))
    result = wf.eval("(X>=2) and (X<4)")
    assert np.array_equal(result.values, np.array([0.0, 0.0, 1.0, 1.0, 0.0]))


def test_descriptive_statistics_match_eviews_moment_conventions():
    values = np.array([1.0, 2.0, 2.0, 5.0, 10.0])
    series = TimeSeries(values, name="X")
    stats = series.describe()

    centered = values - values.mean()
    m2 = np.mean(centered ** 2)
    m3 = np.mean(centered ** 3)
    m4 = np.mean(centered ** 4)

    assert stats["Observations"] == 5
    assert stats["Included observations"] == 5
    assert stats["Mean"] == pytest.approx(values.mean())
    assert stats["Median"] == pytest.approx(2.0)
    assert stats["Std. Dev."] == pytest.approx(np.std(values, ddof=1))
    assert stats["Variance"] == pytest.approx(np.var(values, ddof=1))
    assert stats["Skewness"] == pytest.approx(m3 / m2 ** 1.5)
    assert stats["Kurtosis"] == pytest.approx(m4 / m2 ** 2)
    expected_jb = 5 / 6 * (
        stats["Skewness"] ** 2 + (stats["Kurtosis"] - 3) ** 2 / 4
    )
    assert stats["Jarque-Bera"] == pytest.approx(expected_jb)
    assert stats["Probability"] == pytest.approx(
        __import__("scipy").stats.chi2.sf(expected_jb, 2)
    )


def test_workfile_describe_uses_active_sample_and_missing_counts():
    wf = Workfile()
    wf.add("X", [1.0, np.nan, 3.0, 100.0, 5.0])
    wf.add("Y", [2.0, 4.0, np.nan, 8.0, 10.0])
    wf.set_sample(0, 3)

    table = wf.describe()
    assert table.loc["X", "Observations"] == 4
    assert table.loc["X", "Included observations"] == 3
    assert table.loc["X", "Maximum"] == 100.0
    assert table.loc["Y", "Observations"] == 4
    assert table.loc["Y", "Included observations"] == 3


def test_descriptive_expression_functions_use_current_sample():
    wf = Workfile()
    wf.add("X", [1.0, 2.0, 100.0, 4.0, np.nan])
    wf.set_sample(0, 3)
    assert wf.eval("@mean(X)") == pytest.approx(26.75)
    assert wf.eval("@median(X)") == pytest.approx(3.0)
    assert wf.eval("@max(X)") == pytest.approx(100.0)
    assert wf.eval("@min(X)") == pytest.approx(1.0)
    assert wf.eval("@var(X)") == pytest.approx(np.var([1.0, 2.0, 100.0, 4.0], ddof=0))
    assert wf.eval("@stdev(X)") == pytest.approx(np.std([1.0, 2.0, 100.0, 4.0], ddof=1))
    assert wf.eval("@obs(X)") == pytest.approx(4.0)


def test_workfile_describe_defaults_to_common_sample_and_supports_individual_samples():
    wf = Workfile()
    wf.add("X", [1.0, 2.0, np.nan, 4.0])
    wf.add("Y", [10.0, np.nan, 30.0, 40.0])

    common = wf.describe()
    individual = wf.describe(individual=True)

    assert common.loc["X", "Included observations"] == 2
    assert common.loc["Y", "Included observations"] == 2
    assert individual.loc["X", "Included observations"] == 3
    assert individual.loc["Y", "Included observations"] == 3
    assert wf.stats().equals(common)


def test_descriptive_expression_functions_honor_range_samples():
    wf = Workfile()
    wf.add("X", [1.0, 2.0, 3.0, 100.0, 5.0])
    wf.set_sample(0, 2)
    assert wf.eval("@mean(X)") == pytest.approx(2.0)
    assert wf.eval("@obs(X)") == pytest.approx(3.0)


def test_workfile_correlogram_uses_active_sample_and_eviews_difference_option():
    rng = np.random.default_rng(123)
    values = np.cumsum(rng.normal(size=80))
    wf = Workfile(frequency="M")
    wf.add("Y", values)
    wf.set_sample(10, 69)

    level = wf.correlogram("Y", nlags=8)
    differenced = wf.correlogram("Y", nlags=8, d=1)

    assert level.nobs == 60
    assert differenced.nobs == 59
    assert level.series_name == "Y"
    assert np.isfinite(level.AC).all()
    assert np.isfinite(differenced.AC).all()


def test_adf_defaults_match_eviews_constant_sic_and_schwert_maxlag():
    values = np.cumsum(np.random.default_rng(123).normal(size=221))
    result = adf(TimeSeries(values, name="TBILL"))

    assert result.regression == "c"
    assert result.lag_selection_method.startswith("Automatic based on SIC")
    assert result.max_lag == 14
    assert "Maxlag=14" in result.lag_selection_method or "maxlag=14" in result.lag_selection_method
    assert {"1%", "5%", "10%"}.issubset(result.critical_values)
    assert "MacKinnon" in result.summary()


def test_workfile_adf_uses_active_sample_and_supports_differences():
    values = np.cumsum(np.random.default_rng(9).normal(size=120))
    wf = Workfile(frequency="M")
    wf.add("Y", values)
    wf.set_sample(10, 109)

    level = wf.adf("Y", autolag=None, lags=1)
    diff = wf.adf("Y", autolag=None, lags=1, d=1)

    assert level.nobs == 98
    assert diff.nobs == 97


def test_workfile_uroot_matches_eviews_style_dispatcher():
    values = np.cumsum(np.random.default_rng(5).normal(size=100))
    wf = Workfile(frequency="M")
    wf.add("Y", values)

    adf_result = wf.uroot("Y", test="adf", exog="const", lags=1, autolag=None)
    assert adf_result.test == "Augmented Dickey-Fuller Test"

    with pytest.raises(ValueError):
        wf.uroot("Y", test="kpss", exog="none")


def test_smpl_supports_eviews_multiple_ranges_and_offsets():
    wf = Workfile()
    wf.add("X", np.arange(10.0))
    wf.smpl("smpl 1 3 7 9")
    assert np.array_equal(wf.sample_series("X").values, np.array([0.0, 1.0, 2.0, 6.0, 7.0, 8.0]))
    wf.smpl("smpl @first+1 @last-1")
    assert np.array_equal(wf.sample_series("X").values, np.arange(1.0, 9.0))
