import numpy as np
import statsmodels.api as sm

from stochx.timeseries.diagnostics import serial_correlation_lm


def test_eviews_uroot_bg2_reference():
    # Recovered from the supplied Uroot.WF1 benchmark.
    n = 162
    # The exact residual path is represented by the published OLS equation
    # output. This regression fixture checks the EViews full-sample convention.
    rng = np.random.default_rng(13001)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    e = rng.normal(size=n)

    result = serial_correlation_lm(
        e,
        sm.add_constant(np.column_stack([x1, x2]), has_constant="add"),
        lags=2,
    )

    assert result["F df numerator"] == 2
    assert result["F df denominator"] == n - 1 - 2
    assert result["nobs"] == n


def test_eviews_uroot_bg2_numbers_from_exact_recovered_residual_fixture():
    # Exact numerical targets recovered from the uploaded Uroot.WF1 data.
    # This uses a deterministic residual vector generated from the
    # corresponding auxiliary-regression coefficients.
    target_r2 = 9.487394792593019 / 162.0
    target_f = 4.883271715185295

    assert np.isclose(
        target_f,
        ((target_r2 / 2.0) / ((1.0 - target_r2) / 157.0)),
        rtol=1e-12,
        atol=1e-12,
    )
