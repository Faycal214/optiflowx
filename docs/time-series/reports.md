# Reports, tables and interpretation

## Why report-oriented results matter

EViews users are accustomed to reading analysis through organized views: equation tables, coefficient statistics, diagnostic tables, roots and forecast reports. StochX keeps that report-oriented workflow while making the output programmatically accessible.

## Result objects

Most time-series methods return a result object rather than a bare array. The result can expose:

- a numerical value or vector;
- a summary representation;
- a table representation;
- interpretation text;
- metadata such as the sample, model order or test specification.

For example:

```python
eq = wf.ls("Y C X", name="EQ01")
print(eq.summary())
print(eq.table())
print(eq.interpret())
```

## Correlogram reports

```python
corr = correlogram(y, nlags=12)
print(corr.table())
```

The table includes lag, AC, PAC, Q-statistic, probability, degrees of freedom and confidence bounds according to the StochX convention.

## Root reports

For ARMA-type models, roots are useful for checking stationarity and invertibility:

```python
print(result.roots_report())
```

Do not confuse a root report with an estimation success flag. A model can converge numerically while violating the intended dynamic constraints.

## Interpretation is not a substitute for the raw result

`interpret()` is a convenience layer for communicating the statistical conclusion. The underlying statistic, p-value, coefficients and metadata remain available for reproducible analysis.

A mature scientific-Python documentation page should explain the estimator and its assumptions, then show a compact example and the resulting object rather than hiding the computational steps. StochX follows that model here.

## Exporting reports

Because results are ordinary Python objects/data frames, users can put them into notebooks, CSV/Excel reporting pipelines or application-specific dashboards. This is one of the main advantages over a GUI-only workflow: the same report can be regenerated from the same script and data.

## Benchmarking against EViews

Where an EViews benchmark is part of the StochX test suite, the benchmark defines explicit numerical expectations. This is preferable to saying that all outputs are universally identical: it tells users exactly which conventions have been verified.
