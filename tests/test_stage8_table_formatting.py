import numpy as np
import pytest

from stochx.timeseries import CorrelogramResult, arma, correlogram, white_noise
from stochx.timeseries.table_formatting import format_correlogram, format_correlogram_table


def test_stage8_7_custom_precision_and_missing_token_are_exact():
    result = correlogram(arma(p=1, q=1, phi=[0.45], theta=[0.25], n=120, rng=12), nlags=4, model_df=1)
    formatted = format_correlogram_table(result, precision=2, missing=".")

    assert all(len(value.split(".")[1]) == 2 for value in formatted["AC"])
    assert all(len(value.split(".")[1]) == 2 for value in formatted["Q-Stat"])
    assert formatted["Prob."][0] == "."


def test_stage8_7_fixed_width_text_has_stable_header_and_missing_rendering():
    result = correlogram(arma(p=1, q=1, phi=[0.45], theta=[0.25], n=80, rng=3), nlags=3, model_df=2)
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
    result = correlogram(white_noise(50, rng=1), nlags=4)
    with pytest.raises(TypeError):
        format_correlogram_table(object())
    with pytest.raises(ValueError):
        format_correlogram_table(result, precision=-1)
    with pytest.raises(ValueError):
        format_correlogram_table(result, missing="")


def test_stage8_7_result_remains_the_frozen_public_type():
    result = correlogram(white_noise(50, rng=9), nlags=4)
    assert isinstance(result, CorrelogramResult)
    assert result.lags.tolist() == [1, 2, 3, 4]
