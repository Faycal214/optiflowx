# EViews numerical parity

StochX uses the official EViews Time Series Estimation tutorial as a numerical benchmark in addition to the USTHB course materials used as the methodological specification.

## Phase A benchmark

The benchmark covers the four tutorial equations:

| Equation | EViews specification | Observations |
|---|---|---:|
| EQ01 | `LOG(M1) C LOG(IP) LOG(CPI) TBILL` | 624 |
| EQ02 | `LOG(M1) C LOG(CPI) LOG(CPI(-1)) LOG(CPI(-2))` | 622 |
| EQ02A | `M1 C CPI(0 to -12)` | 612 |
| EQ03 | `LOG(M1) C LOG(M1(-1)) LOG(CPI) LOG(CPI(-1)) LOG(CPI(-2))` | 622 |

The expected EViews values are stored in `tests/fixtures/eviews_phase_a_expected.json`.

The raw `Data.xlsx` benchmark file is intentionally not vendored in the repository. Obtain it from the official EViews Time Series Estimation tutorial and set:

```bash
export STOCHX_EVIEWS_DATA=/path/to/Data.xlsx
```

Then run:

```bash
pytest -q tests/test_eviews_phase_a.py -m parity
```

Without the raw benchmark file the numerical parity test is skipped. A separate range-expansion test runs without external data.

## EViews compatibility rules validated here

- Negative offsets are lags: `X(-1)` means the previous observation.
- Positive offsets are leads: `X(1)` means the next observation.
- Inclusive ranges such as `CPI(0 to -12)` expand to `CPI`, `CPI(-1)`, ..., `CPI(-12)`.
- Equation samples contract automatically after lagged regressors introduce missing observations.
- Equation statistics use EViews' normalized AIC, Schwarz/BIC, Hannan-Quinn, and standard error of regression conventions.

## Planned parity phases

1. **Phase A — basic regressions:** lags, leads, transformations, distributed lags, dynamic OLS.
2. **Phase B — serial correlation:** residual diagnostics, AR/MA/ARMA corrections.
3. **Phase C — heteroskedasticity/ARCH/HAC:** White, ARCH LM, Newey-West/HAC and related outputs.
4. **Phase D — forecasting:** static/dynamic forecasts, forecast evaluation and ARMA/ARIMA forecasting.
5. **Phase E — unit roots:** dedicated EViews ADF/DF workfiles and exact critical-value/statistic parity.

The USTHB course PDFs remain the authoritative specification for the statistical workflow; EViews benchmark packages are the numerical validation layer.
