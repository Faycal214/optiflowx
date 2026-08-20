# EViews-style workflow

## Goal

This page is the bridge for analysts moving an existing EViews time-series workflow into Python. StochX keeps the **order of operations and familiar vocabulary** while making each step explicit Python code.

## The workfile

`Workfile` stores named series, an optional frequency, and the current estimation sample.

```python
from stochx.timeseries import Workfile

wf = Workfile.from_dataframe(frame, frequency="M")
print(wf.info())
print(wf.names())
wf.set_sample(12, 120)
```

A CSV can be loaded directly:

```python
wf = Workfile.from_csv(
    "macro.csv",
    date_column="DATE",
    frequency="M",
)
```

## Series expressions

StochX accepts EViews-inspired expressions for lags, leads and transformations:

| EViews-style expression | Meaning |
|---|---|
| `Y` | current series |
| `Y(-1)` | first lag |
| `Y(1)` | first lead |
| `D(Y)` | first difference |
| `DLOG(Y)` | log difference |
| `LOG(Y)` | natural logarithm |
| `@TREND` | deterministic trend term where supported |

```python
wf.eval("GDP(-1)")
wf.eval("D(GDP)")
wf.generate("DGDP", "D(GDP)")
```

## Equations

The familiar equation specification is kept as a string:

```python
eq = wf.ls("GDP C CONS CONS(-1)", name="EQ01")
print(eq.summary())
print(eq.table())
print(eq.interpret())
```

`C` denotes the intercept in the EViews-style specification. Lags are written directly in the expression.

## Regression with ARMA errors

Serial-correlation terms belong to the **disturbance process** rather than the regressors:

```python
eq = wf.ls(
    "TBILL C LOG(M1) LOG(CPI) LOG(IP) @TREND AR(1) MA(1)",
    name="EQ20",
)
```

The result exposes coefficients, standard errors, t-values, p-values, fit statistics, roots and covariance information where supported.

## Stationarity decision

The same workfile can feed DF/ADF procedures:

```python
from stochx.timeseries import adf

result = adf(wf["GDP"], regression="c", lags=1, autolag=None)
print(result.summary())
print(result.interpret())
```

The point is not to hide the unit-root decision inside a model selector. The deterministic specification, lag order, statistic and critical values remain visible.

## Mapping mindset

A useful migration pattern is:

| EViews habit | StochX habit |
|---|---|
| Workfile | `Workfile` |
| Series view | `TimeSeries` / `Workfile[name]` |
| Generate series | `wf.generate(name, expression)` |
| Lag | `Y(-1)` / `wf.lag("Y")` |
| Difference | `D(Y)` / `wf.diff("Y")` |
| Equation LS | `wf.ls("Y C X")` |
| ADF | `adf(...)` |
| Correlogram | `correlogram(...)` |
| ARMA/ARIMA | `fit_arma`, `fit_arima`, `estimate`, Box–Jenkins workflow |
| Forecast | `forecast_*` helpers |

## What “compatible” means

StochX is **EViews-inspired**, not an attempt to reproduce the EViews application itself. It follows the same statistical workflow and vocabulary where that interface is implemented. Numerical parity is treated as a testable benchmark for specific frozen fixtures rather than as an implicit guarantee for every possible dataset.

## Next steps

Continue with [Data and series](data-series.md), then [Transformations](transforms.md), [Stationarity](stationarity.md), and [Correlation](correlation.md).
