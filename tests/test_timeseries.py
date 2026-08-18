import numpy as np
import pytest

from stochx.timeseries import TimeSeries, acf, pacf


def test_timeseries_metadata_and_summary():
    series = TimeSeries([1.0, 2.0, 3.0], name="Y", frequency="Q")

    assert series.nobs == 3
    assert series.nmissing == 0
    assert series.start is None
    summary = series.summary()
    assert "Y" in summary
    assert "Frequency:             Q" in summary


def test_lag_and_difference():
    series = TimeSeries([1.0, 3.0, 6.0, 10.0], name="Y")

    lagged = series.lag(1)
    assert np.isnan(lagged.values[0])
    assert np.allclose(lagged.values[1:], [1.0, 3.0, 6.0])

    differenced = series.diff()
    assert np.allclose(differenced.values, [2.0, 3.0, 4.0])


def test_acf_at_lag_zero_and_white_noise_shape():
    series = TimeSeries(np.arange(1.0, 21.0), name="Y")
    result = acf(series, nlags=5)

    assert np.isclose(result.values[0], 1.0)
    assert result.lags.tolist() == [0, 1, 2, 3, 4, 5]
    assert np.all(result.upper > 0)
    assert np.all(result.lower < 0)


def test_pacf_at_lag_zero():
    series = TimeSeries(np.arange(1.0, 31.0), name="Y")
    result = pacf(series, nlags=5)

    assert np.isclose(result.values[0], 1.0)
    assert result.lags.tolist() == [0, 1, 2, 3, 4, 5]


def test_constant_series_acf_fails():
    with pytest.raises(ValueError, match="constant series"):
        acf(TimeSeries([1.0] * 10), nlags=3)


def test_missing_values_are_supported_for_lag():
    series = TimeSeries([1.0, np.nan, 3.0], name="Y")
    assert series.nmissing == 1
    lagged = series.lag(1)
    assert np.isnan(lagged.values[0])
    assert np.isclose(lagged.values[2], np.nan, equal_nan=True)
