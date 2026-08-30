# Step 13 — Numerical Validation

Step 13 freezes the statistical implementation from Steps 1–12 and validates observable behavior against captured reference outputs.

Numerical validation is not a legal requirement and does not attempt to reproduce proprietary EViews internals. The target is independent implementation of documented econometric behavior, followed by transparent comparison.

## Validation layers

### Structural

Exact agreement is required for:

    model specification
    sample size
    lag orders
    coefficient labels
    table columns
    model-selection decisions

### Displayed numerical values

When a reference value comes from a displayed EViews table, the tolerance starts at half of the last displayed decimal place, with a configurable relative tolerance.

### Continuous numerical results

Roots, forecasts and other floating-point arrays use feature-specific absolute/relative tolerances. A single global tolerance is intentionally not used.

## Public API

    from stochx.timeseries import (
        ParityReport,
        compare_number,
        compare_array,
        compare_dataframe,
        compare_equation_fixture,
        compare_forecast,
        validate_equations,
        assert_reports_pass,
    )

Multiple equation fixtures can be validated as:

    reports = validate_equations(results, references)
    assert_reports_pass(reports)

Each `ParityReport` records every individual check, its actual and expected value, absolute/relative error, tolerance, and failure reason.

## Existing EViews reference

`tests/fixtures/eviews_phase_b_expected.json` contains captured output for EQ18–EQ21 from the public EViews Time Series Estimation tutorial. The repository does not bundle the EViews `.wf1` file or executable.

Reference scope currently covers:

    ARMA specification
    observation count
    coefficients
    standard errors
    t-statistics
    probabilities
    AIC / SIC / HQ
    inverse AR/MA roots

## Validation command

The EViews-dependent fixture tests remain opt-in:

    STOCHX_EVIEWS_DATA=/path/to/Data.xlsx pytest -q tests/test_eviews_phase_b.py

Without the external workbook, the reference-schema and parity-engine tests still run and verify the validation infrastructure.

## Tolerance policy

The current manifest is stored in `tests/fixtures/parity_manifest.json`.

    OLS display relative tolerance        1e-8
    ARMA display relative tolerance      2e-5
    information-criteria relative        2e-8
    inverse-root relative                2e-2
    default continuous absolute/relative 1e-8

These are starting validation policies, not claims about EViews precision. They are tightened or relaxed only when a reference fixture demonstrates that the difference is numerical rather than a specification mismatch.

## What Step 13 is finding

Failures are classified as:

    missing field / structural mismatch
    label or order mismatch
    displayed-value mismatch
    numerical tolerance mismatch
    missing EViews reference fixture

This lets us fix implementation errors separately from expected backend-level numerical differences.

## Current certification matrix

    Step 01  workfile/sample              reference expansion required
    Step 02  descriptive statistics       reference expansion required
    Step 03  ACF/PACF                     reference expansion required
    Step 04  unit roots                   reference expansion required
    Step 05  ARMA                         EQ18–EQ21 fixture available
    Step 06  ARIMA/SARIMA                 fixture required
    Step 07  equations                    ARMA fixture available
    Step 08  diagnostics                  fixture required
    Step 09  autoarma                     fixture required
    Step 10  forecasting                  fixture required
    Step 11  reports                      EViews text captures required
    Step 12  cointegration/ECM            fixture required

Step 13 therefore begins with the validated ARMA baseline and expands the reference suite feature by feature.

## ARIMA/SARIMA + forecasting reference

A public EViews automatic-ARIMA example is now encoded as tests/fixtures/eviews_arima_forecast_reference.json. The documented case is monthly English/Welsh electricity demand (ELECDMD), using Auto(None/Log), maximum differencing 2, max AR/MA 4, max SAR/SMA 1, periodicity 12, and AIC model selection. EViews reports 100 candidate models and selects a logged, first-differenced (3,3)(1,1) model for the example. The public page documents the estimation sample 2005M01 2014M04 and forecast sample 2014M05 2015M12. 

The public textual documentation does not publish the numerical forecast vector, so the fixture marks forecast numerical capture as pending instead of fabricating expected values. Step 13 therefore validates the documented automatic-ARIMA structure now and numerical coefficients, forecast standard errors, bounds, and other values once an EViews capture is supplied. No EViews workfile or executable belongs in the repository.
