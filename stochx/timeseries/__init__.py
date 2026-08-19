"""Time-series analysis, simulation and forecasting tools.

The public namespace is organized around the USTHB course workflow:
data preparation, smoothing/decomposition, stationarity, identification,
estimation, validation and forecasting, with an EViews-inspired workfile
and equation interface.
"""

from .arma_errors import ErrorProcess, parse_error_terms
from .correlation import ACFResult, PACFResult, acf, pacf
from .correlogram import correlogram
from .decomposition import DecompositionResult, decompose, exponential_smoothing, fisher_seasonality_test, holt, holt_winters, moving_average, seasonal_difference, seasonal_indices, weighted_moving_average
from .diagnostics import TestResult, arch_test, box_pierce, breusch_godfrey, breusch_godfrey_raw, breusch_pagan, durbin_watson_test, jarque_bera, ljung_box, mean_zero_test, normality_ks, redundancy_check, residual_diagnostics, roots_report, variance_ratio_test
from .equation import Equation, EquationResult
from .expression import Expression, ExpressionError, evaluate
from .forecasting import ForecastMetrics, drift_forecast, metrics, naive_forecast, prediction_interval, restore_differences
from .identification import grid_search, identify
from .models import TSResult, estimate, fit_ar, fit_arima, fit_arma, fit_ma, fit_sarima
from .plotting import plot_correlogram, plot_decomposition, plot_forecast, plot_series
from .regression import RegressionResult, ols, trend_terms
from .results import ResultTable, UnifiedResult
from .series import TimeSeries
from .simulation import ar, arma, ma, random_walk, sarma, simulate_process, white_noise
from .stationarity import (
    DF_CRITICAL_VALUES,
    DF_F_CRITICAL_VALUES,
    DF_SPECIFICATIONS,
    SequentialDFResult,
    SpecificationTestResult,
    UnitRootResult,
    adf,
    classify_ts_ds,
    dickey_fuller,
    dickey_fuller_sequential,
    difference,
    kpss_test,
    phillips_perron,
    trend_regression,
)
from .theory import inverse_ar_coefficients, impulse_response, is_invertible_ma, is_stationary_ar, polynomial_roots, process_mean, theoretical_ar_acf, theoretical_ma_acf
from .workfile import Workfile

__all__ = [
    "TimeSeries", "Workfile", "Expression", "ExpressionError", "evaluate", "Equation", "EquationResult", "UnifiedResult", "ResultTable",
    "ErrorProcess", "parse_error_terms",
    "ACFResult", "PACFResult", "acf", "pacf", "correlogram",
    "DF_SPECIFICATIONS", "DF_CRITICAL_VALUES", "DF_F_CRITICAL_VALUES", "UnitRootResult", "SpecificationTestResult", "SequentialDFResult",
    "adf", "dickey_fuller", "kpss_test", "phillips_perron", "dickey_fuller_sequential", "classify_ts_ds", "difference", "trend_regression",
    "DecompositionResult", "moving_average", "weighted_moving_average", "exponential_smoothing", "holt", "holt_winters", "decompose", "seasonal_difference", "seasonal_indices", "fisher_seasonality_test",
    "TSResult", "fit_ar", "fit_ma", "fit_arma", "fit_arima", "fit_sarima", "estimate",
    "identify", "grid_search",
    "TestResult", "durbin_watson_test", "breusch_godfrey", "breusch_godfrey_raw", "box_pierce", "ljung_box", "jarque_bera", "mean_zero_test", "normality_ks", "variance_ratio_test", "breusch_pagan", "arch_test", "residual_diagnostics", "roots_report", "redundancy_check",
    "ForecastMetrics", "metrics", "prediction_interval", "restore_differences", "naive_forecast", "drift_forecast",
    "RegressionResult", "ols", "trend_terms",
    "white_noise", "ar", "ma", "arma", "random_walk", "sarma", "simulate_process",
    "polynomial_roots", "is_stationary_ar", "is_invertible_ma", "process_mean", "impulse_response", "inverse_ar_coefficients", "theoretical_ma_acf", "theoretical_ar_acf",
    "plot_series", "plot_correlogram", "plot_forecast", "plot_decomposition",
]
