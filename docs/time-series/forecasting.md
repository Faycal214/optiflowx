# Forecasting parity

EViews treats forecasting as an equation procedure with an explicit forecast sample, static/dynamic solution method, optional structural forecast, forecast standard errors, coefficient-uncertainty control, MA backcast control, and forecast evaluation. 

## Equation API

Dynamic forecast:

    eq.forecast(start=120, end=140, dynamic=True)

Static forecast:

    eq.fit(start=120, end=140)

Structural forecast:

    eq.forecast(start=120, end=140, structural=True)

EViews defines Static as a sequence of one-step-ahead forecasts using actual lagged dependent values where available; Dynamic recursively uses forecasted lagged dependent values. With ARMA terms, both include the forecasted residual process by default; Structural suppresses ARMA terms. 

## Forecast output

StochX normalizes forecast output to:

    Forecast
    Std. Error
    Lower
    Upper

EViews forecast standard errors represent forecast uncertainty and may include coefficient uncertainty. Its forecast dialog provides a switch to omit coefficient uncertainty. 

## ARMA forecasting

For AR errors, EViews uses actual or forecasted lagged residual information according to Static/Dynamic mode. MA errors require presample innovation values and a backcast procedure. The current StochX API exposes `ma_backcast` but exact EViews MA pre-sample recursion remains fixture-dependent. 

## Transformations

The automatic ARIMA path supports `tform='auto'`, `tform='none'`, and `tform='log'`. Log forecasts are back-transformed to the original scale, including the interval endpoints. EViews notes that nonlinear dependent-variable expressions require special interval transformation rather than simply symmetrically adding a standard error on the original scale. 

## Forecast evaluation

EViews reports RMSE, MAE, MAPE and Theil inequality coefficient and also provides bias, variance and covariance proportions. StochX exposes these through `eq.forecast_evaluation(forecast, actual)`. 

## Parity boundary

Implemented: forecast sample controls, static/dynamic API, structural ARMA switch, forecast-error standard-error output, log back-transformation, forecast evaluation, and explicit MA backcast option.

Still requiring direct EViews fixtures: exact static ARMA residual recursion, exact MA presample values, exact coefficient-uncertainty-off standard errors, exact nonlinear-expression interval normalization, actual-versus-NA fill behavior, and exact finite-sample forecast interval equality.