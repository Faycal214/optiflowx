"""Time-series analysis, simulation and forecasting tools.

The public namespace is organized around the USTHB course workflow:
data preparation, smoothing/decomposition, stationarity, identification,
estimation, validation and forecasting, with an EViews-inspired workfile
and equation interface.
"""

from .arma_errors import ErrorProcess, parse_error_terms
from .correlation import ACFResult, PACFResult, acf, pacf
from .correlogram import CorrelogramResult, LjungBoxResult, correlogram, ljung_box as correlogram_ljung_box
from .decomposition import DecompositionResult, decompose, exponential_smoothing, fisher_seasonality_test, holt, holt_winters, moving_average, seasonal_difference, seasonal_indices, weighted_moving_average
from .diagnostics import TestResult, arch_test, box_pierce, breusch_godfrey, breusch_godfrey_raw, breusch_pagan, durbin_watson_test, jarque_bera, ljung_box, mean_zero_test, normality_ks, redundancy_check, residual_correlogram, residual_diagnostics, residual_diagnostics_correlogram, roots_report, variance_ratio_test
from .equation import Equation, EquationResult
from .expression import Expression, ExpressionError, evaluate
from .forecasting import ForecastMetrics, drift_forecast, metrics, naive_forecast, prediction_interval, restore_differences
from .identification import BoxJenkinsIdentificationResult, grid_search, identify, identify_box_jenkins
from .interpretation import interpret_correlogram
from .models import TSResult, estimate, fit_ar, fit_arima, fit_arma, fit_ma, fit_sarima
from .plotting import plot_correlogram, plot_decomposition, plot_eviews_correlogram, plot_forecast, plot_series
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
from .table_formatting import format_correlogram, format_correlogram_table
from .theory import inverse_ar_coefficients, impulse_response, is_invertible_ma, is_stationary_ar, polynomial_roots, process_mean, theoretical_ar_acf, theoretical_ma_acf
from .workfile import Workfile
from .box_jenkins_estimation import BoxJenkinsEstimationResult, EstimatedCandidate, estimate_box_jenkins_candidates
from .box_jenkins_validation import CandidateValidation, BoxJenkinsValidationResult, validate_box_jenkins_candidates
from .box_jenkins_selection import BoxJenkinsSelectionResult, select_box_jenkins_model

__all__ = [
    "TimeSeries", "Workfile", "Expression", "ExpressionError", "evaluate", "Equation", "EquationResult", "UnifiedResult", "ResultTable",
    "ErrorProcess", "parse_error_terms",
    "ACFResult", "PACFResult", "acf", "pacf", "CorrelogramResult", "LjungBoxResult", "correlogram", "correlogram_ljung_box", "ljung_box", "format_correlogram_table", "format_correlogram", "interpret_correlogram",
    "DF_SPECIFICATIONS", "DF_CRITICAL_VALUES", "DF_F_CRITICAL_VALUES", "UnitRootResult", "SpecificationTestResult", "SequentialDFResult",
    "adf", "dickey_fuller", "kpss_test", "phillips_perron", "dickey_fuller_sequential", "classify_ts_ds", "difference", "trend_regression",
    "DecompositionResult", "moving_average", "weighted_moving_average", "exponential_smoothing", "holt", "holt_winters", "decompose", "seasonal_difference", "seasonal_indices", "fisher_seasonality_test",
    "TSResult", "fit_ar", "fit_ma", "fit_arma", "fit_arima", "fit_sarima", "estimate",
    "BoxJenkinsIdentificationResult", "identify_box_jenkins", "identify", "grid_search",
    "EstimatedCandidate", "BoxJenkinsEstimationResult", "estimate_box_jenkins_candidates",
    "CandidateValidation", "BoxJenkinsValidationResult", "validate_box_jenkins_candidates",
    "BoxJenkinsSelectionResult", "select_box_jenkins_model",
    "TestResult", "durbin_watson_test", "breusch_godfrey", "breusch_godfrey_raw", "box_pierce", "ljung_box", "jarque_bera", "mean_zero_test", "normality_ks", "variance_ratio_test", "breusch_pagan", "arch_test", "residual_correlogram", "residual_diagnostics_correlogram", "residual_diagnostics", "roots_report", "redundancy_check",
    "ForecastMetrics", "metrics", "prediction_interval", "restore_differences", "naive_forecast", "drift_forecast",
    "RegressionResult", "ols", "trend_terms",
    "white_noise", "ar", "ma", "arma", "random_walk", "sarma", "simulate_process",
    "polynomial_roots", "is_stationary_ar", "is_invertible_ma", "process_mean", "impulse_response", "inverse_ar_coefficients", "theoretical_ar_acf", "theoretical_ma_acf",
    "plot_series", "plot_correlogram", "plot_eviews_correlogram", "plot_forecast", "plot_decomposition",
]
