"""EViews-style time-series API gallery."""

from __future__ import annotations

import numpy as np

from stochx.timeseries import (
    Workfile,
    acf,
    autoarma,
    correlogram,
    difference,
    fit_ar,
    fit_arima,
    fit_arma,
    fit_ma,
    johansen,
    ols,
)


def main() -> None:
    y = np.cumsum(np.random.default_rng(7).normal(size=120))
    x = np.random.default_rng(8).normal(size=120)

    wf = Workfile(frequency="Q")
    wf.add("Y", y)
    wf.add("X", x)

    eq = wf.ls("Y C X", name="EQ1")
    print(eq.summary())
    print(eq.forecast())
    print(correlogram(y, nlags=12).table())
    print(acf(y, nlags=12).values)

    print(difference(y, order=1))
    print(fit_ar(y, p=2).summary())
    print(fit_ma(y, q=2).summary())
    print(fit_arma(y, p=1, q=1).summary())
    print(fit_arima(y, order=(1, 1, 1)).summary())
    print(autoarma(y).summary())

    # Public functional OLS/Equation entry points are both available.
    print(ols(y, np.column_stack([np.ones(len(y)), x])).summary())


if __name__ == "__main__":
    main()
