"""End-to-end Stage 11 local-level state-space workflow example."""

import numpy as np

from stochx.timeseries import run_local_level_workflow


y = np.array([1.0, 1.2, np.nan, 1.3, 1.4, 1.25, 1.5, 1.55])

result = run_local_level_workflow(
    y,
    diagnostic_lags=2,
    alpha=0.10,
    forecast_steps=3,
)

print("Filtered states:")
print(result.filter_result.filtered_state[:, 0])
print("Smoothed states:")
print(result.smoother.smoothed_state[:, 0])
print("Forecast:")
print(result.forecast.forecast[:, 0])
print("Forecast lower bound:")
print(result.forecast.lower[:, 0])
print("Forecast upper bound:")
print(result.forecast.upper[:, 0])
print("Adequate model:", result.adequacy.adequate)
