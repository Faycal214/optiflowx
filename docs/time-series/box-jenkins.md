# Box–Jenkins workflow

## The methodology

The Box–Jenkins approach is a sequence rather than a single estimator:

**identification → estimation → validation → selection → forecasting**.

StochX exposes each stage separately so a user can inspect what happened and reuse intermediate results.

## Identification

```python
from stochx.timeseries import identify_box_jenkins

ident = identify_box_jenkins(
    y,
    d=1,
    nlags=24,
    max_p=3,
    max_q=3,
)
print(ident.candidate_orders)
```

Identification uses the differencing order supplied by the analyst and the ACF/PACF evidence to build a deterministic candidate set.

## Estimation

```python
from stochx.timeseries import estimate_box_jenkins_candidates

estimation = estimate_box_jenkins_candidates(
    y,
    ident.candidate_orders,
)
```

Every candidate retains auditable estimation metadata, including success/convergence state, parameters, standard errors, roots, likelihood and information criteria.

## Validation

```python
from stochx.timeseries import validate_box_jenkins_candidates

validation = validate_box_jenkins_candidates(
    estimation,
    lags=12,
    alpha=0.05,
)
```

The mandatory adequacy rule is residual serial adequacy using the requested Ljung–Box lags. Optional mean-zero, normality and ARCH checks can make eligibility stricter.

## Selection

```python
from stochx.timeseries import select_box_jenkins_model

selection = select_box_jenkins_model(
    validation,
    criterion="aic",
    tie_tolerance=1e-8,
)
print(selection.selected_order)
print(selection.rationale)
```

Selection is deterministic. Ties do not depend on dictionary order or platform iteration order.

## Forecasting

```python
from stochx.timeseries import forecast_box_jenkins

forecast = forecast_box_jenkins(selection, steps=12, alpha=0.05)
print(forecast.forecast)
print(forecast.lower)
print(forecast.upper)
```

The result preserves forecast metadata and the original index when it can infer a valid future frequency.

## Report-oriented workflow

For an EViews user, the conceptual mapping is:

```text
View / Correlogram
        ↓
Identify ARIMA orders
        ↓
Estimate candidates
        ↓
Inspect residuals
        ↓
Compare AIC/SC/HQ
        ↓
Forecast
```

The StochX advantage is that each view is a Python object rather than an ephemeral GUI state, so the complete analysis can be rerun and tested.

## Failure semantics

Candidate estimation failures do not silently disappear. They remain represented in the result set and are excluded from selection only when the validation/selection contract says they are ineligible. If no adequate model remains, selection reports that state explicitly rather than inventing a fallback.

See [Diagnostics](diagnostics.md) before treating a selected candidate as an accepted model.
