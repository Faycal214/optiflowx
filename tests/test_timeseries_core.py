import numpy as np

# existing file content preserved above/below

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
