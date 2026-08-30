import numpy as np
import pandas as pd
from stochx.timeseries import TimeSeries, Workfile, cointreg, ecm, engle_granger, johansen, vecm

def _cointegrated(n=220, seed=1201):
    rng=np.random.default_rng(seed); x=np.cumsum(rng.normal(size=n)); y=1.5+0.8*x+rng.normal(scale=0.7,size=n); return x,y

def test_engle_granger():
    x,y=_cointegrated(); r=engle_granger(TimeSeries(y,name='Y'),TimeSeries(x,name='X')); assert r.method=='Engle-Granger'; assert np.isfinite(r.statistic)

def test_cointreg_methods():
    x,y=_cointegrated(220,1202)
    for m in ['fmols','ccr','dols']:
        r=cointreg(y,x,method=m,leads=2,lags=2); assert 'X1' in r.params.index; assert np.isfinite(r.long_run_variance)

def test_ecm():
    x,y=_cointegrated(220,1203); r=ecm(TimeSeries(y,name='Y'),TimeSeries(x,name='X'),lags=2); assert 'EC(-1)' in r.table().index

def test_johansen():
    x,y=_cointegrated(220,1204); r=johansen(pd.DataFrame({'Y':y,'X':x}),k_ar_diff=1); assert 0<=r.rank<=2; assert 'Trace Statistic' in r.rank_table()

def test_vecm_and_workfile():
    x,y=_cointegrated(220,1205); wf=Workfile(); wf.add('Y',y); wf.add('X',x); r=wf.johansen(['Y','X']); assert r.variables==('Y','X'); v=wf.vecm(['Y','X'],rank=1); assert v.rank==1
