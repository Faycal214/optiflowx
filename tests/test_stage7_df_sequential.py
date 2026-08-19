from dataclasses import replace

import numpy as np

import stochx.timeseries.stationarity as stationarity
from stochx.timeseries import SequentialDFResult, TimeSeries


# Keep an unpatched reference because several tests monkeypatch stationarity.adf.
_REAL_ADF = stationarity.adf


def _base_result(regression: str, decision: str, alpha: float = 0.05):
    base = _REAL_ADF(
        np.linspace(1.0, 80.0, 80),
        regression=regression,
        lags=0,
        autolag=None,
        alpha=alpha,
    )
    return replace(base, decision=decision, conclusion=f"forced {regression}: {decision}")


def test_stage7_common_lag_order_is_used_for_all_models(monkeypatch):
    calls = []
    original_adf = stationarity.adf

    def recording_adf(*args, **kwargs):
        calls.append((kwargs.get("regression"), kwargs.get("lags"), kwargs.get("autolag")))
        return original_adf(*args, **kwargs)

    monkeypatch.setattr(stationarity, "adf", recording_adf)
    result = stationarity.dickey_fuller_sequential(
        TimeSeries(np.random.default_rng(17).normal(size=300), name="Y"),
        max_lags=2,
        autolag=None,
        alpha=0.05,
    )

    assert isinstance(result, SequentialDFResult)
    assert [r.regression for r in result.tests] == ["ct", "c", "n"]
    assert [r.lags for r in result.tests] == [2, 2, 2]
    assert [c[1] for c in calls] == [2, 2, 2]
    assert result.lag_order == 2


def test_stage7_model3_rejects_and_beta_is_significant(monkeypatch):
    original_fit = stationarity._fit_df_regression

    def fake_adf(y, *, regression, lags=None, autolag=None, alpha=0.05):
        return _base_result(regression, "reject" if regression == "ct" else "fail_to_reject", alpha)

    def fake_fit(x, regression, lags):
        fitted = original_fit(np.asarray(x), regression, lags)
        if regression == "ct":
            fitted["tvalues"] = np.array([0.0, 3.0, -5.0])
            fitted["params"] = np.array([1.0, 2.0, -0.5])
        return fitted

    monkeypatch.setattr(stationarity, "adf", fake_adf)
    monkeypatch.setattr(stationarity, "_fit_df_regression", fake_fit)

    result = stationarity.dickey_fuller_sequential(
        np.arange(1.0, 81.0), max_lags=0, autolag=None
    )

    assert result.selected.regression == "ct"
    assert result.specification_tests[0].name == "Model 3 trend test"
    assert result.specification_tests[0].decision == "reject"
    assert "deterministic trend" in result.nature


def test_stage7_model3_rejects_beta_insignificant_then_model2_checks_c(monkeypatch):
    original_fit = stationarity._fit_df_regression

    def fake_adf(y, *, regression, lags=None, autolag=None, alpha=0.05):
        return _base_result(
            regression,
            "reject" if regression in {"ct", "c"} else "fail_to_reject",
            alpha,
        )

    def fake_fit(x, regression, lags):
        fitted = original_fit(np.asarray(x), regression, lags)
        if regression == "ct":
            fitted["tvalues"] = np.array([0.0, 0.2, -5.0])
        elif regression == "c":
            fitted["tvalues"] = np.array([3.0, -5.0])
        return fitted

    monkeypatch.setattr(stationarity, "adf", fake_adf)
    monkeypatch.setattr(stationarity, "_fit_df_regression", fake_fit)

    result = stationarity.dickey_fuller_sequential(
        np.arange(1.0, 81.0), max_lags=0, autolag=None
    )

    assert result.selected.regression == "c"
    assert [s.name for s in result.specification_tests] == [
        "Model 3 trend test",
        "Model 2 constant test",
    ]
    assert result.specification_tests[0].decision == "fail_to_reject"
    assert result.specification_tests[1].decision == "reject"
    assert "constant" in result.nature


def test_stage7_model3_unit_root_not_rejected_uses_f3_then_can_continue_to_model2(monkeypatch):
    def fake_adf(y, *, regression, lags=None, autolag=None, alpha=0.05):
        return _base_result(regression, "fail_to_reject", alpha)

    def fake_joint_f(x, regression, lags, alpha):
        return stationarity.SpecificationTestResult(
            name="Model 3 joint F test" if regression == "ct" else "Model 2 joint F test",
            null_hypothesis="H0: joint deterministic specification",
            alternative_hypothesis="H1: joint specification alternative",
            statistic=99.0 if regression == "ct" else 1.0,
            critical_value=6.0 if regression == "ct" else 4.0,
            decision="reject" if regression == "ct" else "fail_to_reject",
            alpha=alpha,
            source="focused regression test",
        )

    monkeypatch.setattr(stationarity, "adf", fake_adf)
    monkeypatch.setattr(stationarity, "_joint_f_test", fake_joint_f)

    result = stationarity.dickey_fuller_sequential(
        np.arange(1.0, 81.0), max_lags=0, autolag=None
    )

    assert result.selected.regression == "ct"
    assert result.specification_tests[0].name == "Model 3 joint F test"
    assert result.specification_tests[0].decision == "reject"
    assert result.nature.startswith("I(1)")


def test_stage7_model3_f3_not_rejected_then_model2_f2_can_retain_integrated_specification(monkeypatch):
    def fake_adf(y, *, regression, lags=None, autolag=None, alpha=0.05):
        return _base_result(regression, "fail_to_reject", alpha)

    def fake_joint_f(x, regression, lags, alpha):
        return stationarity.SpecificationTestResult(
            name="Model 3 joint F test" if regression == "ct" else "Model 2 joint F test",
            null_hypothesis="H0: joint deterministic specification",
            alternative_hypothesis="H1: joint specification alternative",
            statistic=1.0 if regression == "ct" else 99.0,
            critical_value=6.0 if regression == "ct" else 4.0,
            decision="fail_to_reject" if regression == "ct" else "reject",
            alpha=alpha,
            source="focused regression test",
        )

    monkeypatch.setattr(stationarity, "adf", fake_adf)
    monkeypatch.setattr(stationarity, "_joint_f_test", fake_joint_f)

    result = stationarity.dickey_fuller_sequential(
        np.arange(1.0, 81.0), max_lags=0, autolag=None
    )

    assert [s.name for s in result.specification_tests] == [
        "Model 3 joint F test",
        "Model 2 joint F test",
    ]
    assert result.selected.regression == "c"
    assert "DS candidate" in result.nature


def test_stage7_course_f2_f3_critical_values_are_nonstandard_and_specification_specific():
    f2_100, source2 = stationarity._course_f_critical("F2", 100, 0.05)
    f3_100, source3 = stationarity._course_f_critical("F3", 100, 0.05)
    assert f2_100 == 4.71
    assert f3_100 == 6.49
    assert source2.startswith("USTHB DF table")
    assert source3.startswith("USTHB DF table")
    assert f2_100 != f3_100


def test_stage7_interpretation_reports_terminal_course_decision(monkeypatch):
    def fake_adf(y, *, regression, lags=None, autolag=None, alpha=0.05):
        return _base_result(regression, "fail_to_reject", alpha)

    monkeypatch.setattr(stationarity, "adf", fake_adf)
    monkeypatch.setattr(
        stationarity,
        "_joint_f_test",
        lambda x, regression, lags, alpha: stationarity.SpecificationTestResult(
            name="Model 3 joint F test" if regression == "ct" else "Model 2 joint F test",
            null_hypothesis="H0: joint deterministic specification",
            alternative_hypothesis="H1: joint specification alternative",
            statistic=1.0,
            critical_value=6.0 if regression == "ct" else 4.0,
            decision="fail_to_reject",
            alpha=alpha,
            source="focused regression test",
        ),
    )

    result = stationarity.dickey_fuller_sequential(
        np.arange(1.0, 81.0), max_lags=0, autolag=None
    )
    text = result.interpret()
    assert "unit-root null" in text
    assert "terminal specification" in text
