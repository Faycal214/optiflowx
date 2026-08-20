# Data and `TimeSeries`

## Why a series object?

A time series is more than a NumPy vector: it has observations, optional labels, a frequency and a name. StochX keeps those pieces together through `TimeSeries` and `Workfile`.

## Creating a series

```python
from stochx.timeseries import TimeSeries

x = TimeSeries(
    [10.0, 10.5, 11.0, 11.4],
    index=["2024M01", "2024M02", "2024M03", "2024M04"],
    name="GDP",
    frequency="M",
)
```

For tabular data, prefer a `Workfile`:

```python
wf = Workfile.from_dataframe(frame, frequency="M")
series = wf["GDP"]
```

## What to inspect first

Before modeling, check:

```python
print(series.nobs)
print(series.values)
print(series.index)
print(series.describe())
```

The descriptive report is intended to answer the first EViews-style questions: how many observations are available, how many are included, what is the mean and dispersion, and what are the minimum and maximum values.

## Missing values

StochX keeps missing observations explicit instead of silently dropping positions from the series object. This matters for lags, differences, diagnostics and state-space models because the time axis must remain aligned.

For filtering, missing values are handled at the scalar observation level; see the [state-space guide](state-space.md).

## Sampling

A workfile can keep a current estimation sample:

```python
wf.set_sample(20, 120)
y_sample = wf.sample_series("GDP")
```

Date-like labels can also be used when the workfile is indexed:

```python
wf.set_sample("2010-01-01 2020-12-01")
```

The sample is part of the workflow state, so generated series and equation estimation use the same observation window unless a method specifies otherwise.

## Data preparation checklist

1. Verify the time index and frequency.
2. Check missing values.
3. Inspect descriptive statistics.
4. Plot the raw series.
5. Generate only the transformations needed for the model.
6. Decide the estimation sample before comparing models.

This separation makes the later model results reproducible and easier to audit.
