import numpy as np
import pytest

from stochx.timeseries import CorrelogramResult, TimeSeries, correlogram
from stochx.timeseries.table_formatting import format_correlogram, format_correlogram_table


def _arma11(phi: float, theta: float, n: int, seed: int) -> TimeSeries:
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=n)
    values = np.empty(n)
    values[0] = eps[0]
    for t in range(1, n):
        values[t] = phi * values[t - 1] + eps[t] + theta * eps[t - 1]
    return TimeSeries(values, name="ARMA11")


def _white_noise(n: int, seed: int) -> TimeSeries:
    return TimeSeries(np.random.default_rng(seed).normal(size=n), name="WN")


def test_stage8_7_custom_precision_and_missing_token_are_exact():
    result = correlogram(_arma11(0.45, 0.25, 120, 12), nlags=4, model_df=1)
    formatted = format_correlogram_table(result, precision=2, missing=".")

    assert all(len(value.split(".")[1]) == 2 for value in formatted["AC"])
    assert all(len(value.split(".")[1]) == 2 for value in formatted["Q-Stat"])
    assert formatted["Prob."][0] == "."


def test_stage8_7_fixed_width_text_has_stable_header_and_missing_rendering():
    result = correlogram(_arma11(0.45, 0.25, 80, 3), nlags=3, model_df=2)
    text = format_correlogram(result)
    lines = text.splitlines()

    assert lines[0].split() == [
        "Lag", "AC", "PAC", "Q-Stat", "Prob.", "DF",
        "AC", "Lower", "AC", "Upper", "PAC", "Lower", "PAC", "Upper",
    ]
    assert "NA" in text
    assert len(lines) == 4
    assert lines[1].lstrip().startswith("1 ")


def test_stage8_7_formatters_validate_inputs():
    result = correlogram(_white_noise(50, 1), nlags=4)
    with pytest.raises(TypeError):
        format_correlogram_table(object())
    with pytest.raises(ValueError):
        format_correlogram_table(result, precision=-1)
    with pytest.raises(ValueError):
        format_correlogram_table(result, missing="")


def test_stage8_7_result_remains_the_frozen_public_type():
    result = correlogram(_white_noise(50, 9), nlags=4)
    assert isinstance(result, CorrelogramResult)
    assert result.lags.tolist() == [1, 2, 3, 4]
