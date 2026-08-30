import numpy as np

from stochx.timeseries import Workfile


def test_eviews_main_equation_report_has_expected_structure():
    rng = np.random.default_rng(1101)
    x = rng.normal(size=160)
    y = 1.25 + 0.8 * x + rng.normal(size=160)

    wf = Workfile()
    wf.add("Y", y)
    wf.add("X", x)
    eq = wf.ls("Y C X", name="EQ11")

    text = eq.show()
    assert "Dependent Variable: Y" in text
    assert "Method: Least Squares" in text
    assert "Included observations:" in text
    assert "Coefficient" in text
    assert "Std. Error" in text
    assert "t-Statistic" in text
    assert "Prob." in text
    assert "R-squared" in text
    assert "Akaike info criterion" in text
    assert "Schwarz criterion" in text
    assert "Durbin-Watson stat" in text


def test_eviews_coefficient_table_column_order_and_labels():
    rng = np.random.default_rng(1102)
    y = rng.normal(size=170)
    x = rng.normal(size=170)

    wf = Workfile()
    wf.add("Y", y)
    wf.add("X", x)
    eq = wf.ls("Y C X AR(1)", name="EQ12")

    table = eq.coefficient_output()
    assert list(table.columns) == [
        "Coefficient", "Std. Error", "t-Statistic", "Prob."
    ]
    assert "C" in table.index
    assert "X" in table.index
    assert "AR(1)" in table.index
    assert "SIGMASQ" in table.index


def test_eviews_named_views_render_as_text():
    rng = np.random.default_rng(1103)
    y = rng.normal(size=180)
    wf = Workfile()
    wf.add("Y", y)
    eq = wf.ls("Y C", name="EQ13")

    assert "Coefficient Covariance Matrix" in eq.view_text("covariance")
    assert "Histogram-Normality" in eq.view_text("histogram-normality")
    assert "Correlogram - Q-statistics" in eq.view_text("correlogram-q-statistics")


def test_eviews_combined_report_can_include_covariance_and_diagnostics():
    rng = np.random.default_rng(1104)
    x = rng.normal(size=180)
    y = 2 + 0.4 * x + rng.normal(size=180)

    wf = Workfile()
    wf.add("Y", y)
    wf.add("X", x)
    eq = wf.ls("Y C X", name="EQ14")

    report = eq.report(
        include_covariance=True,
        include_diagnostics=True,
        diagnostic_lags=6,
    )
    assert "Coefficient Covariance Matrix" in report
    assert "Residual Diagnostics" in report
    assert "R-squared" in report


def test_eviews_arma_report_uses_eviews_parameter_names():
    rng = np.random.default_rng(1105)
    e = rng.normal(size=210)
    y = np.zeros(210)
    for t in range(1, 210):
        e[t] = 0.6 * e[t - 1] + e[t]
        y[t] = 1.0 + e[t]

    wf = Workfile()
    wf.add("Y", y)
    eq = wf.ls("Y C AR(1) MA(1)", name="EQ15")
    text = eq.show()
    assert "AR(1)" in text
    assert "MA(1)" in text
    assert "SIGMASQ" in text
