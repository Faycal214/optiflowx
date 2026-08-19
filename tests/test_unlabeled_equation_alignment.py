import numpy as np

from stochx.timeseries.workfile import Workfile


def test_unlabeled_y_c_x_keeps_all_observations_aligned():
    x = np.arange(1.0, 101.0)
    y = 3.0 + 2.0 * x + np.sin(x)
    wf = Workfile()
    wf.add("X", x)
    wf.add("Y", y)

    assert wf["Y"].index is None
    assert wf["X"].index is None

    result = wf.ls("Y C X", name="EQ_UNLABELED")

    assert result.nobs == 100
    assert np.isfinite(result.params.to_numpy()).all()
    assert result.table().index.tolist() == ["C", "X"]


def test_unlabeled_lagged_regressor_keeps_only_genuine_missing_lag():
    x = np.arange(1.0, 101.0)
    y = 4.0 + 1.5 * x
    wf = Workfile()
    wf.add("X", x)
    wf.add("Y", y)

    result = wf.ls("Y C X(-1)", name="EQ_UNLABELED_LAG")

    assert result.nobs == 99
    assert "X(-1)" in result.table().index


def test_unlabeled_phase_a_lag_range_estimation_aligns_columns():
    x = np.arange(1.0, 30.0)
    wf = Workfile()
    wf.add("M1", 100.0 + x)
    wf.add("CPI", 20.0 + 0.2 * x)

    result = wf.ls("M1 C CPI(0 to -12)", name="EQ02A_UNLABELED")

    assert result.nobs == 17
    assert result.table().index.tolist() == [
        "C",
        "CPI",
        "CPI(-1)",
        "CPI(-2)",
        "CPI(-3)",
        "CPI(-4)",
        "CPI(-5)",
        "CPI(-6)",
        "CPI(-7)",
        "CPI(-8)",
        "CPI(-9)",
        "CPI(-10)",
        "CPI(-11)",
        "CPI(-12)",
    ]
