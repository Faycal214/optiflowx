# Stage 9 — Box–Jenkins

## Baseline

Stage 9 is branched from the frozen Stage 8 commit:

`31806c2ee25c7ab8e89f1549e5bd59040a65179e`

The Stage 8 numerical and public API contract is frozen. Stage 9 must not change the AC/PAC, Ljung–Box, confidence-band, `CorrelogramResult`, or Stage 8 residual-correlogram contracts.

## Existing building blocks

StochX already provides AR/MA/ARMA/ARIMA/SARIMA estimators, order comparison, identification helpers, residual diagnostics, forecasting, and the frozen residual correlogram. Stage 9 will build the Box–Jenkins workflow around these components rather than replacing them.

## Stage 9 sequence

### 9.1 Conventions and workflow contract

Freeze the identification → estimation → diagnostic validation → model selection → forecasting workflow, sample/lag conventions, candidate-order bounds, and stopping rules.

### 9.2 Identification

Integrate differencing/stationarity status with ACF/PACF evidence and explicit candidate ARMA/ARIMA orders.

### 9.3 Candidate estimation

Estimate candidate models using the existing model layer and expose comparable likelihood, parameter count, AIC, Schwarz/BIC, and HQ measures.

### 9.4 Residual validation

Use the frozen Stage 8 residual correlogram and Ljung–Box results as the canonical whiteness diagnostic, with coefficient/root/stability checks.

### 9.5 Model selection

Apply transparent parsimony rules when multiple candidates are adequate, keeping diagnostics auditable.

### 9.6 Forecasting

Connect the selected model to forecast generation, intervals, and forecast evaluation without changing existing forecast APIs.

### 9.7 Numerical fixtures and regression tests

Add deterministic candidate-model and Box–Jenkins workflow fixtures and focused regression tests.

### 9.8 CI and freeze

Run the complete matrix, freeze Stage 9 only after all local and CI checks are green.
