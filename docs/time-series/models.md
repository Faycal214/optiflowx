# Time-series models

StochX exposes the standard univariate families used throughout the course workflow.

## AR(p)

The autoregressive model is

$$y_t=c+\phi_1y_{t-1}+\cdots+\phi_py_{t-p}+\varepsilon_t.$$

```python
from stochx.timeseries import fit_ar

result = fit_ar(y, p=2)
print(result.summary())
```

An AR model is useful when the current value can be explained largely by its own recent history. Stationarity depends on the roots of the AR polynomial.

## MA(q)

A moving-average model uses past shocks:

$$y_t=c+\varepsilon_t+\theta_1\varepsilon_{t-1}+\cdots+\theta_q\varepsilon_{t-q}.$$

```python
from stochx.timeseries import fit_ma
result = fit_ma(y, q=2)
```

Invertibility is checked through the MA roots.

## ARMA(p,q)

ARMA combines both structures:

$$\phi(B)y_t=c+\theta(B)\varepsilon_t.$$

```python
from stochx.timeseries import fit_arma
result = fit_arma(y, p=1, q=1)
```

## ARIMA(p,d,q)

ARIMA applies ARMA structure after $d$ ordinary differences. Use it when the original series is not stationary but its differenced representation is.

```python
from stochx.timeseries import fit_arima
result = fit_arima(y, order=(1, 1, 1))
```

## SARIMA

SARIMA extends ARIMA with seasonal orders. The parameterization separates non-seasonal and seasonal dynamics so that the final specification remains auditable.

```python
from stochx.timeseries import fit_sarima
result = fit_sarima(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
```

## What every fitted result should answer

A useful model report should make the following visible:

- estimated coefficients;
- standard errors;
- t-values and p-values where applicable;
- log-likelihood and information criteria;
- residuals;
- stationarity/invertibility information;
- forecast behavior;
- diagnostic tests.

## Don't stop at the fit

A model is not selected because its estimation converged. The StochX workflow continues through residual validation and deterministic selection. See [Diagnostics](diagnostics.md) and [Box–Jenkins](box-jenkins.md).

## ARIMA and SARIMA

EViews specifies integer differencing separately from the ARMA disturbance structure. A direct ARIMA workflow therefore uses an ARIMA(p,d,q) state-space likelihood after applying the integer integration order, while seasonal AR and MA terms are multiplicative seasonal polynomials. EViews supports AR, SAR, MA and SMA terms and allows the user to explicitly list the desired lags. citeturn506442search4turn825961search0

StochX exposes:

```python
fit_arima(y, 1, 1, 1)
fit_sarima(y, (1, 1, 1), (1, 1, 1, 12))
wf.sarima("Y", (1, 1, 1), (1, 1, 1, 12))
```

The seasonal period can be inferred from a dated monthly/quarterly workfile or supplied explicitly. EViews defaults seasonal periodicity to the number of observations per year for a dated workfile. citeturn506442search1

The direct ARIMA/SARIMA estimation path uses state-space maximum likelihood with explicit BFGS and OPG covariance, matching the EViews ML contract established in Step 5. EViews documents ARIMA ML as evaluation through the Kalman filter prediction-error decomposition. citeturn506442search6turn875983search23

Forecasting exposes both recursive/dynamic and one-step/static modes through the result object:

```python
result.forecast(12, dynamic=False)
result.forecast(12, dynamic=True)
```

The EViews forecasting tutorial explicitly distinguishes static and dynamic forecasting and covers ARMA-equation forecasting. citeturn875983search0

### Step 6 numerical-parity boundary

The ARIMA/SARIMA architecture and EViews defaults are implemented. Exact equality of finite-sample likelihood values, coefficients, roots and forecast intervals still requires captured EViews ARIMA/SARIMA output fixtures, because Kalman initialization and likelihood initialization are part of the numerical contract. EViews documents the Kalman prediction-error likelihood but the exact EViews state initialization must still be benchmarked against actual EViews output before claiming bit-level parity. citeturn506442search6
