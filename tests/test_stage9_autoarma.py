import numpy as np
from stochx.timeseries import TimeSeries, Workfile, autoarma

def test_eviews_autoarma_defaults_match_documented_defaults():
    y = TimeSeries(np.random.default_rng(1001).normal(size=90), name='Y')
    result = autoarma(y, max_diff=0)
    assert result.criterion == 'AIC'
    assert {'p','d','q','P','D','Q','AIC','SIC','HQ','converged','included'} <= set(result.table().columns)

def test_eviews_autoarma_uses_successive_kpss_for_differencing():
    y = TimeSeries(np.cumsum(np.random.default_rng(1002).normal(size=120)), name='Y')
    result = autoarma(y, max_diff=2)
    assert result.differencing_order in {0,1,2}
    assert result.kpss_history[0]['d'] == 0

def test_eviews_autoarma_defaults_to_four_ar_and_ma_terms():
    y = TimeSeries(np.random.default_rng(1003).normal(size=90), name='Y')
    result = autoarma(y, max_diff=0)
    table = result.table()
    assert table['p'].max() == 4 and table['q'].max() == 4
    assert table['P'].max() == 0 and table['Q'].max() == 0

def test_workfile_autoarma_uses_active_sample_and_monthly_period():
    wf = Workfile(frequency='M')
    wf.add('Y', np.random.default_rng(1004).normal(size=90))
    wf.set_sample(10, 89)
    result = wf.autoarma('Y', max_diff=0)
    assert result.seasonal_period == 12
    assert len(result.table()) == 25

def test_autoarma_log_requires_positive_observations():
    y = TimeSeries([1,2,3,4,5]*20, name='Y')
    result = autoarma(y, max_diff=0, max_ar=1, max_ma=1, tform='log')
    assert result.transformation == 'log'