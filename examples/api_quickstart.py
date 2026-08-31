"""Minimal StochX time-series quickstart."""

import numpy as np

from stochx.timeseries import Workfile, correlogram, fit_arima


rng = np.random.default_rng(42)
y = np.cumsum(rng.normal(size=120))

wf = Workfile(frequency="Q")
wf.add("Y", y)

equation = wf.ls("Y C")
print(equation.summary())

model = fit_arima(y, order=(1, 1, 1))
print(model.summary())

print(correlogram(y, nlags=12).table())
