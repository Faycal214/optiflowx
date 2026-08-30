# StochX Architecture

StochX is designed as a **time-series and econometrics library with an executable methodological workflow**.

The central rule is:

> **The mathematical definition and econometric convention come first. The Python object represents the concept. The documentation explains the mapping.**

## 1. Layers

### Layer A — course mathematics

The supplied USTHB material defines the main methodological sequence:

```text
observed series
→ transformations / stationarization
→ deterministic specification
→ ACF / PACF
→ identification
→ estimation
→ validation / testing
→ selection
→ forecasting
```

The mathematical scope includes stationary processes, deterministic and stochastic non-stationarity, TS/DS processes, unit roots, DF/ADF procedures, AR/MA/ARMA models, seasonal structures, ARIMA-family models, parameter estimation, validation, diagnostics and forecasting.

### Layer B — computational models

The Python implementation represents these concepts directly through objects and functions such as:

```python
Workfile(...)
TimeSeries(...)
acf(...)
adf(...)
fit_ar(...)
fit_ma(...)
fit_arma(...)
fit_arima(...)
fit_sarima(...)
ols(...)
```

Higher-level workflows compose these primitives into Box–Jenkins and forecasting pipelines.

### Layer C — EViews compatibility

EViews compatibility is treated as an explicit engineering layer. Familiar operations include workfiles, samples, series expressions, lags, differences, deterministic terms, equation specifications, reports, diagnostics and forecasts.

Where a reference EViews output exists, it is recorded as a numerical fixture with documented conventions.

## 2. Package organization

The public implementation lives under:

```text
stochx/
└── timeseries/
    ├── workfile.py
    ├── series.py
    ├── expression.py
    ├── regression.py
    ├── equation.py
    ├── decomposition.py
    ├── correlation.py
    ├── correlogram.py
    ├── stationarity.py
    ├── models.py
    ├── diagnostics.py
    ├── box_jenkins_*.py
    ├── forecasting.py
    ├── statespace*.py
    ├── theory.py
    ├── plotting.py
    └── results.py
```

There is intentionally **no general stochastic-process or simulation subsystem** in the package.

## 3. Public API design

Public functions should expose:

- the mathematical meaning of the operation;
- relevant econometric assumptions;
- parameters and their interpretation;
- result objects containing the information needed for interpretation;
- deterministic behavior wherever a numerical convention is part of the contract.

## 4. Numerical conventions

The package is numerical rather than symbolic:

- NumPy/SciPy provide finite-dimensional numerical computation;
- pandas provides indexed data structures;
- statsmodels may be used as an implementation component where its conventions are compatible;
- StochX-specific conventions must remain explicit and testable.

The external library is never treated as authoritative merely because it provides a function with the same name.

## 5. Testing philosophy

Tests are organized around:

1. mathematical correctness;
2. EViews numerical parity;
3. edge cases and missing observations;
4. public API contracts;
5. documentation and release integrity.

A parity fixture is evidence for a specific dataset, specification and convention. It is not a claim of universal bit-for-bit equality across software.
