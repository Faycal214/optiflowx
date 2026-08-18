"""Time-series analysis, simulation and forecasting tools.

The public namespace is deliberately organized around the workflow used in the
provided USTHB time-series course: data preparation, smoothing/decomposition,
stationarity, identification, estimation, validation and forecasting.
"""

from .correlation import ACFResult, PACFResult, acf, pacf
from .correlogram import correlogram
from .decomposition import (
    DecompositionResult,
    decompose,
    exponential_smoothing,
    fisher_seasonality_test,
    holt,
    holt_winters,
    moving_average,
    seasonal_difference,
    seasonal_indices,
    weighted_moving_average,
)
from .diagnostics import (
    TestResult,
    arch_test,
    box_pierce,
    breusch_pagan,
    durbin_watson_test,
    jarque_bera,
    ljung_box,
    mean_zero_test,
    normality_ks,
    redundancy_check,
    residual_diagnostics,
    roots_report,
    variance_ratio_test,
)
from .forecasting import ForecastMetrics, drift_forecast, metrics, naive_forecast, prediction_interval, restore_differences
from .identification import grid_search, identify
from .models import TSResult, estimate, fit_ar, fit_arima, fit_arma, fit_ma, fit_sarima
from .plotting import plot_correlogram, plot_decomposition, plot_forecast, plot_series
from .regression import RegressionResult, ols, trend_terms
from .series import TimeSeries
from .simulation import ar, arma, ma, random_walk, sarma, simulate_process, white_noise
from .stationarity import UnitRootResult, adf, classify_ts_ds, dickey_fuller_sequential, difference, kpss_test, phillips_perron, trend_regression
from .workfile import Workfile

__all__ = [
    "TimeSeries", "Workfile", "ACFResult", "PACFResult", "acf", "pacf", "correlogram",
    "UnitRootResult", "adf", "kpss_test", "phillips_perron", "dickey_fuller_sequential", "classify_ts_ds", "difference", "trend_regression",
    "DecompositionResult", "moving_average", "weighted_moving_average", "exponential_smoothing", "holt", "holt_winters", "decompose", "seasonal_difference", "seasonal_indices", "fisher_seasonality_test",
    "TSResult", "fit_ar", "fit_ma", "fit_arma", "fit_arima", "fit_sarima", "estimate",
    "identify", "grid_search",
    "TestResult", "durbin_watson_test", "box_pierce", "ljung_box", "jarque_bera", "mean_zero_test", "normality_ks", "variance_ratio_test", "breusch_pagan", "arch_test", "residual_diagnostics", "roots_report", "redundancy_check",
    "ForecastMetrics", "metrics", "prediction_interval", "restore_differences", "naive_forecast", "drift_forecast",
    "RegressionResult", "ols", "trend_terms",
    "white_noise", "ar", "ma", "arma", "random_walk", "sarma", "simulate_process",
    "plot_series", "plot_correlogram", "plot_forecast", "plot_decomposition",
]
