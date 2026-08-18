"""EViews-inspired StochX workflow for Time Series coursework."""

from __future__ import annotations

import numpy as np

from stochx.timeseries import Workfile, adf, estimate


if __name__ == "__main__":
    x = np.arange(1.0, 101.0)
    y = 5.0 + 0.8 * x + np.sin(x)
    wf = Workfile()
    wf.add("X", x)
    wf.add("Y", y)

    print(wf.info())
    print(wf.eval("X(-1)"))
    print(wf.generate("DX", "D(X)").summary())

    equation = wf.ls("Y C X", name="EQ01")
    print(equation.summary())
    print(equation.interpret())

    unit_root = adf(wf["Y"], regression="c", lags=1, autolag=None)
    print(unit_root.summary())
    print(unit_root.interpret())

    ar1 = estimate(wf["Y"], p=1)
    print(ar1.summary())
    print(ar1.interpret())
