"""EViews-style automatic ARIMA selection."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
import numpy as np
import pandas as pd
import statsmodels.api as sm
from .models import fit_arima, fit_sarima
from .stationarity import kpss_test
from .series import TimeSeries

@dataclass(frozen=True)
class AutoARIMAResult:
    candidates: pd.DataFrame
    selected_order: tuple[int, int, int, int, int, int]
    selected_result: object
    differencing_order: int
    seasonal_period: int
    criterion: str
    kpss_history: tuple[dict[str, object], ...]
    transformation: str

    @property
    def selected(self): return self.selected_result
    def table(self): return self.candidates.copy()
    def forecast(self, steps: int, **kwargs): return self.selected_result.forecast(steps=steps, **kwargs)

def _series(y):
    return y if isinstance(y, TimeSeries) else TimeSeries(np.asarray(list(y.values if hasattr(y, "values") else y), dtype=float), name=getattr(y, "name", "Y"))

def _difference(x, d):
    return np.diff(x, n=d) if d else x.copy()

def _auto_log_choice(values: np.ndarray) -> str:
    """Apply EViews Auto(None/Log) heteroskedasticity rule."""
    if np.any(values <= 0):
        return "none"
    dy = np.diff(values)
    ly = np.log(values)
    dly = np.diff(ly)
    r1 = sm.OLS(dy**2, sm.add_constant(values[1:], has_constant='add')).fit()
    r2 = sm.OLS(dly**2, sm.add_constant(ly[1:], has_constant='add')).fit()
    t1 = float(np.asarray(r1.tvalues)[1])
    t2 = float(np.asarray(r2.tvalues)[1])
    return "log" if abs(t2) < abs(t1) else "none"
def _choose_d(y, max_diff, alpha):
    x = y.copy(); hist=[]
    for d in range(max_diff + 1):
        r=kpss_test(x, regression="c", alpha=alpha)
        hist.append({"d":d,"statistic":r.statistic,"pvalue":r.pvalue,"decision":r.decision})
        if r.decision=="fail_to_reject" or d==max_diff: return d, x, tuple(hist)
        x=np.diff(x)
    raise RuntimeError("differencing selection failed")

def autoarma(y, *, max_diff=2, max_ar=4, max_ma=4, max_sar=0, max_sma=0, periods=None, select="aic", kpss_sig=0.05, nonconv=False, tform="auto", name=None):
    s=_series(y); vals=np.asarray(s.values,dtype=float); vals=vals[np.isfinite(vals)]
    if vals.size < 12: raise ValueError("autoarma requires at least 12 finite observations")
    if tform not in {"auto", "none", "log"}:
        raise ValueError("tform must be auto, none, or log in the certified path")
    if tform == "auto":
        tform = _auto_log_choice(vals)
    if tform == "log":        if np.any(vals<=0): raise ValueError("log transformation requires strictly positive observations")
        vals=np.log(vals)
    if periods is None:
        key=str(s.frequency).upper() if s.frequency is not None else ""
        periods={"M":12,"MONTHLY":12,"Q":4,"QUARTERLY":4,"W":52,"WEEKLY":52,"D":7,"DAILY":7}.get(key,1)
    d, transformed, hist=_choose_d(vals,max_diff,kpss_sig)
    rows=[]; fitted={}
    for p,q,P,Q in product(range(max_ar+1),range(max_ma+1),range(max_sar+1),range(max_sma+1)):
        key=(p,d,q,P,0,Q)
        try:
            if P or Q: r=fit_sarima(TimeSeries(vals,name=name or s.name), (p,d,q), (P,0,Q,int(periods)))
            else: r=fit_arima(TimeSeries(vals,name=name or s.name), p,d,q)
            st=r.statistics(); conv=bool(getattr(r.result,"mle_retvals",{}).get("converged",True))
            if not conv and not nonconv: raise ValueError("non-converged model excluded")
            rows.append({"p":p,"d":d,"q":q,"P":P,"D":0,"Q":Q,"LogLik":st["Log likelihood"],"AIC":st["Akaike info criterion"],"SIC":st["Schwarz criterion"],"HQ":st["Hannan-Quinn criterion"],"converged":conv,"included":True,"error":None})
            fitted[key]=r
        except Exception as exc:
            rows.append({"p":p,"d":d,"q":q,"P":P,"D":0,"Q":Q,"LogLik":np.nan,"AIC":np.nan,"SIC":np.nan,"HQ":np.nan,"converged":False,"included":False,"error":str(exc)})
    frame=pd.DataFrame(rows)
    crit={"aic":"AIC","sic":"SIC","hq":"HQ"}.get(select.lower())
    if crit is None: raise ValueError("select must be aic, sic, or hq")
    eligible=frame.loc[frame.included & np.isfinite(frame[crit])].sort_values([crit,"p","q","P","Q"])
    if eligible.empty: raise ValueError("no eligible converged candidate models")
    row=eligible.iloc[0]; order=tuple(int(row[k]) for k in ("p","d","q","P","D","Q"))
    return AutoARIMAResult(frame.sort_values(crit).reset_index(drop=True),order,fitted[order],d,int(periods),crit.upper(),hist,tform)
