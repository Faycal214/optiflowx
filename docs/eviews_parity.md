# EViews compatibility and numerical parity

StochX uses **two references for its time-series layer**:

1. the USTHB course material defines the statistical methodology, notation and decision workflow;
2. official EViews Time Series Estimation examples provide numerical benchmarks for the subset of workflows that are explicitly tested.

This distinction is important. StochX is intended to feel familiar to EViews users, but it is implemented as a native Python library. We therefore claim parity only where a benchmark exists and passes.

## What is intentionally familiar?

The following vocabulary and operations are deliberately recognizable:

| EViews concept | StochX |
|---|---|
| workfile | `Workfile` |
| sample | `Workfile.set_sample(...)` |
| series | `TimeSeries`, `Workfile[name]` |
| lag | `Y(-1)` / `Workfile.lag()` |
| lead | `Y(1)` |
| difference | `D(Y)` |
| log difference | `DLOG(Y)` |
| logarithm | `LOG(Y)` |
| intercept | `C` in supported equation specifications |
| trend | `@TREND` in supported equation specifications |
| equation LS | `Workfile.ls(...)` |
| ADF / DF | `adf(...)`, `dickey_fuller(...)` |
| correlogram | `correlogram(...)` |
| ARMA error | `AR(...)`, `MA(...)` |
| equation report | `summary()`, `table()`, `statistics()` |
| interpretation | `interpret()` |
| roots | `roots_report()` |
| forecast | model-specific `forecast_*` functions |

## A real EViews-to-StochX translation

An EViews-style equation

```text
Y C X X(-1)
```

becomes:

```python
eq = wf.ls("Y C X X(-1)", name="EQ01")
```

A transformed series:

```text
series DY = D(Y)
```

becomes:

```python
wf.generate("DY", "D(Y)")
```

A simple unit-root view becomes:

```python
result = adf(wf["Y"], regression="c", lags=1, autolag=None)
print(result.summary())
```

The point is not syntactic mimicry for its own sake. Keeping familiar commands reduces the amount of methodological translation an EViews user has to perform.

## Phase A — basic equation parity

The repository contains a Phase A benchmark based on the official EViews Time Series Estimation tutorial. The expected values are stored in:

```text
tests/fixtures/eviews_phase_a_expected.json
```

The raw `Data.xlsx` file is intentionally not vendored. Obtain it from the official tutorial and set:

```bash
export STOCHX_EVIEWS_DATA=/path/to/Data.xlsx
```

Then run:

```bash
pytest -q tests/test_eviews_phase_a.py -m parity
```

Without the external workbook, the numerical parity test is skipped. The repository still validates the expression/range expansion behavior independently.

## Validated EViews conventions

The current benchmark documents these conventions explicitly:

- negative offsets are lags;
- positive offsets are leads;
- inclusive distributed-lag ranges expand to every included lag;
- lagged regressors contract the usable equation sample automatically;
- equation reports expose the information criteria and standard-error conventions used by the benchmark.

## Output-report philosophy

The goal is **the same analysis story**, not a screenshot clone.

A StochX result should let an EViews user answer the same questions from Python:

- Which observations were estimated?
- Which coefficients were obtained?
- How precise are those coefficients?
- What are the fit statistics?
- Are the residuals adequate?
- What are the roots/stability diagnostics?
- What are the forecasts and uncertainty bounds?

`summary()`, `table()`, `statistics()`, `roots_report()`, `interpret()` and forecast result objects are the programmatic report layer.

## Parity roadmap

The repository currently organizes parity as explicit validation phases:

1. basic regressions: transformations, lags, leads and distributed lags;
2. serial correlation: AR/MA/ARMA corrections and residual diagnostics;
3. heteroskedasticity/ARCH/HAC;
4. forecasting and forecast evaluation;
5. unit-root and DF/ADF benchmarks.

Each phase should be considered complete only when the corresponding numerical fixture is committed and tested.

## Important limitation

An EViews benchmark is evidence for a **specific dataset, specification and numerical convention**. It is not evidence that every possible dataset will match bit-for-bit across software. This is why StochX keeps parity fixtures separate from the general API contract.
