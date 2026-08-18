import numpy as np

from stochx.timeseries import (
    TimeSeries,
    adf,
    ar,
    correlogram,
    difference,
    estimate,
    fisher_seasonality_test,
    identify,
    ljung_box,
    moving_average,
    random_walk,
    seasonal_difference,
    white_noise,
)
from stochx.timeseries.forecasting import metrics
from stochx.timeseries.workfile import Workfile


def test_series_transformations_and_correlation():
    y = TimeSeries(np.arange(1.0, 101.0), name="Y", frequency="M")
    assert y.nobs == 100
    assert y.diff().nobs == 99
    assert y.log().nobs == 100
    assert y.lag(1).nmissing == 1
    assert y.acf(12).lags[-1] == 12
    assert y.pacf(12).lags[-1] == 12


def test_workfile_and_sample_workflow():
    y = TimeSeries(np.arange(1.0, 51.0), name="Y", frequency="M")
    dy = TimeSeries(np.r_[np.nan, np.diff(y.values)], name="D_Y", frequency="M")
    wf = Workfile()
    wf.add("Y", y.values)
    wf.add("D_Y", dy.values)
    wf.set_sample(5, 40)
    assert wf.nobs == 50
    assert wf.sample.start == 5
    assert wf.sample.stop == 41


def test_simulation_and_identification_pipeline():
    y = ar(1, [0.6], 250, rng=0)
    ident = identify(y, nlags=12)
    assert ident["ACF"].nobs == 250
    result = estimate(y, p=1, d=0, q=0)
    assert result.params is not None
    corr = correlogram(y, nlags=12)
    assert {"AC", "PAC", "Q-Stat", "Prob."}.issubset(corr.columns)


def test_stationarity_and_forecast_metrics():
    y = random_walk(120, rng=0)
    result = adf(y, regression="c", lags=1, autolag=None)
    assert np.isfinite(result.statistic)
    assert result.pvalue is not None
    assert {"1%", "5%", "10%"}.issubset(result.critical_values)
    assert "Augmented Dickey-Fuller Test" in result.summary()
    assert difference(y, 1).nobs == 119
    ma = moving_average(y, 5)
    assert ma.nobs == 120
    err = metrics([1, 2, 3], [1, 1, 4])
    assert err.rmse > 0


def test_seasonality_and_residual_test():
    seasonal = TimeSeries(np.tile([10.0, 12.0, 15.0, 11.0], 20), name="S", frequency="Q")
    seasonal_d = seasonal_difference(seasonal, 4)
    assert seasonal_d.nobs == 76
    fisher = fisher_seasonality_test(seasonal, 4)
    assert fisher["reject_seasonality_null"]
    lb = ljung_box(white_noise(200, rng=0), lags=12)
    assert np.isfinite(lb.statistic)
