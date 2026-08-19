import numpy as np

import stochx.timeseries.stationarity as stationarity


def test_stage7_df_critical_values_use_course_table_rows():
    values, source = stationarity._course_df_critical("ct", 91)
    assert values == {"1%": -4.04, "5%": -3.45, "10%": -3.15}
    assert source == "USTHB DF table, n=100"

    values_250, source_250 = stationarity._course_df_critical("c", 250)
    assert values_250["5%"] == -2.88
    assert source_250 == "USTHB DF table, n=250"


def test_stage7_f2_f3_use_same_course_table_size_convention():
    f2, source2 = stationarity._course_f_critical("F2", 91, 0.05)
    f3, source3 = stationarity._course_f_critical("F3", 91, 0.05)
    assert f2 == 4.71
    assert f3 == 6.49
    assert source2 == "USTHB DF table, n=100"
    assert source3 == "USTHB DF table, n=100"

    f2_inf, source_inf = stationarity._course_f_critical("F2", 251, 0.05)
    assert f2_inf == 4.59
    assert source_inf == "USTHB DF table, n=∞"


def test_stage7_whitening_selector_chooses_smallest_parsimonious_p(monkeypatch):
    calls = []

    def fake_whitening(x, *, regression, lags, test_lags, alpha):
        calls.append((regression, lags, test_lags, alpha))
        return (lags >= 2, 0.10 if lags >= 2 else 0.01, test_lags)

    monkeypatch.setattr(stationarity, "_residual_whitening", fake_whitening)
    x = np.arange(100.0)
    lag, method = stationarity._selected_common_lag(
        x,
        max_lags=5,
        autolag="AIC",
        alpha=0.05,
        whitening_lags=12,
    )

    assert lag == 2
    assert "minimum p=2" in method
    assert [entry[1] for entry in calls] == [0, 1, 2]


def test_stage7_sequential_default_uses_whitening_policy(monkeypatch):
    calls = []

    def fake_whitening(x, *, regression, lags, test_lags, alpha):
        calls.append(lags)
        return (lags == 1, 0.08 if lags == 1 else 0.02, test_lags)

    monkeypatch.setattr(stationarity, "_residual_whitening", fake_whitening)
    x = np.arange(80.0)
    lag, method = stationarity._selected_common_lag(
        x,
        max_lags=4,
        autolag="AIC",
        alpha=0.05,
        whitening_lags=8,
    )
    assert lag == 1
    assert calls == [0, 1]
    assert "Ljung-Box whitening" in method
