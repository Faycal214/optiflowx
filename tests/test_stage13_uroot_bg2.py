import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stochx.timeseries.diagnostics import serial_correlation_lm


@pytest.mark.parametrize("lags", [1, 2, 4])
def test_breusch_godfrey_preserves_full_sample_and_reports_f_df(lags):
    rng = np.random.default_rng(13001 + lags)
    n = 162
    e = rng.normal(size=n)
    x = pd.DataFrame({
        "C": np.ones(n),
        "GDP": rng.normal(size=n),
        "CS(-1)": rng.normal(size=n),
    })

    result = serial_correlation_lm(e, x.to_numpy(), lags=lags)

    assert result["nobs"] == n
    assert result["F df numerator"] == lags
    assert result["F df denominator"] == n - x.shape[1] - lags


def test_eviews_uroot_bg2_same_data_reference():
    csv_path = os.environ.get(
        "STOCHX_UROOT_CSV",
        "validation_data/uroot_CS_GDP_extracted_with_presample.csv",
    )
    if not Path(csv_path).exists():
        pytest.skip(f"Uroot validation data not available: {csv_path}")

    frame = pd.read_csv(Path(csv_path))
    required = {"date", "CS", "GDP"}
    assert required <= set(frame.columns)

    frame["CS_lag"] = frame["CS"].shift(1)
    frame = frame.loc[
        (frame["date"] >= "1948Q3") & (frame["date"] <= "1988Q4"),
        ["CS", "GDP", "CS_lag"],
    ].dropna()

    # The EViews equation is CS C GDP CS(-1).
    import statsmodels.api as sm

    fit = sm.OLS(
        frame["CS"].to_numpy(),
        sm.add_constant(frame[["GDP", "CS_lag"]].to_numpy(), has_constant="add"),
    ).fit()

    result = serial_correlation_lm(
        fit.resid,
        fit.model.exog,
        lags=2,
    )

    assert result["nobs"] == 162
    assert result["F df numerator"] == 2
    assert result["F df denominator"] == 157
    assert np.isclose(result["LM statistic"], 9.487394792593019, rtol=1e-10, atol=1e-10)
    assert np.isclose(result["p-value"], 0.008706395605758378, rtol=1e-10, atol=1e-12)
    assert np.isclose(result["F-statistic"], 4.883271715185295, rtol=1e-10, atol=1e-10)
    assert np.isclose(result["F p-value"], 0.008761368086309455, rtol=1e-10, atol=1e-12)
