import numpy as np
import pytest

from stochx.timeseries import correlogram
from stochx.timeseries.correlation import acf, pacf


def test_acf_uses_one_common_complete_observation_sample():
    x = np.arange(1.0, 11.0)
    missing = x.copy()
    missing[0] = np.nan
    missing[5] = np.nan
    missing[-1] = np.nan

    clean = acf(x[[1, 2, 3, 4, 6, 7, 8, 9]], nlags=3)
    result = acf(missing, nlags=3)

    assert result.nobs == 8
    assert result.missing_count == 3
    assert np.allclose(result.values, clean.values, rtol=1e-13, atol=1e-13)


def test_pacf_uses_same_effective_sample_and_missing_count():
    missing = np.arange(1.0, 11.0)
    missing[[0, 5, 9]] = np.nan
    result = pacf(missing, nlags=3)

    assert result.nobs == 7
    assert result.missing_count == 3


def test_correlogram_carries_effective_nobs_and_missing_audit_fields():
    x = np.arange(1.0, 21.0)
    x[[1, 7, 18]] = np.nan
    result = correlogram(x, nlags=6)

    assert result.nobs == 17
    assert result.missing_count == 3
    assert result.nlags == 6
    assert len(result.lags) == 6
    assert len(result.Q_Stat) == 6
    assert len(result.Prob) == 6
    assert np.all(np.isfinite(result.Q_Stat))

    table = result.table()
    assert len(table) == 6
    assert table["Lag"].tolist() == [1, 2, 3, 4, 5, 6]


def test_correlogram_keeps_lag_order_and_uses_only_genuine_missing_drop():
    x = np.arange(1.0, 8.0)
    x[2] = np.nan
    result = correlogram(x, nlags=10)

    # Six usable observations imply at most five displayed lags.
    assert result.nobs == 6
    assert result.missing_count == 1
    assert result.nlags == 5
    assert result.lags.tolist() == [1, 2, 3, 4, 5]


def test_all_missing_and_insufficient_nonmissing_values_raise():
    with pytest.raises(ValueError, match="at least two"):
        acf([np.nan, np.nan], nlags=1)

    with pytest.raises(ValueError, match="at least two"):
        pacf([1.0, np.nan], nlags=1)


def test_infinite_values_are_rejected_not_treated_as_missing():
    with pytest.raises(ValueError, match="infinite"):
        correlogram([1.0, 2.0, np.inf, 4.0], nlags=2)
