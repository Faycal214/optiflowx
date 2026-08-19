import numpy as np
import pandas as pd
import pytest

from stochx.timeseries import (
    CorrelogramResult,
    arma,
    correlogram,
    format_correlogram,
    format_correlogram_table,
    white_noise,
)


def test_stage8_7_ordinary_formatting_is_deterministic_and_numeric_contract_is_unchanged():
    result = correlogram(white_noise(90, rng=4), nlags=5, model_df=0)

    numeric_before = result.table().copy()
    formatted = format_correlogram_table(result)

    assert list(formatted.columns) == [
        "Lag", "AC", "PAC", "Q-Stat", "Prob.", "DF",
        "AC Lower", "AC Upper", "PAC Lower", "PAC Upper",
    ]
    assert formatted.shape == (5, 10)
    assert formatted["Lag"].tolist() == ["1", "2", "3", "4", "5"]
    assert formatted["DF"].tolist() == ["1", "2", "3", "4", "5"]
    assert all(len(value.split("." )[1]) == 4 for value in formatted["AC"])
    assert all(len(value.split("." )[1]) == 4 for value in formatted["Q-Stat"])

    # Presentation must not mutate or round the numerical projection.
    pd.testing.assert_frame_equal(result.table(), numeric_before)
    assert np.issubdtype(result.table()["AC"].dtype, np.floating)
    assert np.issubdtype(result.table()["Q-Stat"].dtype, np.floating)


def test_stage8_7_residual_formatting_renders_undefined_probabilities_as_NA():
    result = correlogram(arma(p=1, q=1, phi=[0.45], theta=[0.25], n=160, rng=7), nlags=6, model_df=2)
    formatted = format_correlogram_table(result)

    assert formatted["DF"].tolist()[:4] == ["-1", "0", "1", "2"]
    assert formatted["Prob."][:2].tolist() == ["NA", "NA"]
    assert formatted["Prob."][2:].map(lambda value: value != "NA").all()
    assert all(len(value.split("." )[1]) == 4 for value in formatted["PAC"])


def test_stage8_7_custom_precision_and_missing_token_are_exact():
    result = correlogram(arma(p=1, q=1, phi=[0.45], theta=[0.25], n=120, rng=12), nlags=4, model_df=1)
    formatted = format_correlogram_table(result, precision=2, missing=".")

    assert all(len(value.split("." )[1]) == 2 for value in formatted["AC"])
    assert all(len(value.split("." )[1]) == 2 for value in formatted["Q-Stat"])
    assert formatted["Prob."][0] == "."


def test_stage8_7_fixed_width_text_has_stable_header_and_missing_rendering():
    result = correlogram(arma(p=1, q=1, phi=[0.45], theta=[0.25], n=80, rng=3), nlags=3, model_df=2)
    text = format_correlogram(result)
    lines = text.splitlines()

    assert lines[0] == "Lag      AC       PAC      Q-Stat  Prob.  DF  AC Lower  AC Upper  PAC Lower  PAC Upper"
    assert "NA" in text
    assert len(lines) == 4
    assert lines[1].startswith("  1 ")


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
    assert result.AC is result.ac
    assert result.Q_Stat is result.q_stat
    assert result.Prob is result.pvalues
