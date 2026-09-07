"""Time-series analysis and forecasting tools.

The public namespace is organized around the USTHB course workflow:
data preparation, smoothing/decomposition, stationarity, identification,
estimation, validation and forecasting, with an EViews-inspired workfile
and equation interface.
"""

from .arma_errors import ErrorProcess, parse_error_terms
from .cointegration import CointegrationTestResult, CointegratingRegressionResult, ECMResult, JohansenResult, VECMResult, cointreg, engle_granger, phillips_ouliaris, johansen, vecm, ecm
from .correlation import ACFResult, PACFResult, acf, pacf
from .correlogram import CorrelogramResult, LjungBoxResult, correlogram, ljung_box as correlogram_ljung_box
from .decomposition import DecompositionResult, decompose, exponential_smoothing, fisher_seasonality_test, holt, holt_winters, moving_average, seasonal_difference, seasonal_indices, weighted_moving_average
from .diagnostics import TestResult, arch_test, box_pierce, breusch_godfrey, breusch_godfrey_raw, breusch_pagan, durbin_watson_test, jarque_bera, ljung_box, mean_zero_test, normality_ks, redundancy_check, residual_correlogram, residual_diagnostics, residual_diagnostics_correlogram, roots_report, variance_ratio_test
from .equation import Equation, EquationResult
from .expression import Expression, ExpressionError, evaluate
from .forecasting import ForecastMetrics, drift_forecast, metrics, naive_forecast, prediction_interval, restore_differences
from .identification import BoxJenkinsIdentificationResult, grid_search, identify, identify_box_jenkins
from .interpretation import interpret_correlogram
from .auto_arima import AutoARIMAResult, autoarma
from .models import TSResult, estimate, fit_ar, fit_arma, fit_arima, fit_ma, fit_sarima
from .plotting import plot_correlogram, plot_decomposition, plot_eviews_correlogram, plot_forecast, plot_series
from .regression import RegressionResult, ols, trend_terms
from .results import ResultTable, UnifiedResult
from .series import TimeSeries
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
from .box_jenkins_forecasting import BoxJenkinsForecastResult, forecast_box_jenkins
from .statespace import KalmanFilterResult, LinearStateSpace, kalman_filter, local_level_filter
from .statespace_smoothing import KalmanSmootherResult, kalman_smoother
from .statespace_forecasting import KalmanForecastResult, kalman_forecast
from .statespace_estimation import LocalLevelEstimateResult, estimate_local_level
from .statespace_diagnostics import KalmanInnovationDiagnosticsResult, kalman_innovation_diagnostics
from .statespace_adequacy import StateSpaceAdequacyResult, state_space_adequacy
from .statespace_workflow import StateSpaceWorkflowResult, run_local_level_workflow


class _CallableNamesView:
    """Live, iterable view that preserves the historical ``wf.names()`` call."""
    def __init__(self, workfile: Workfile) -> None:
        self._workfile = workfile
    def __iter__(self):
        return iter(self._workfile.series)
    def __len__(self) -> int:
        return len(self._workfile.series)
    def __contains__(self, item: object) -> bool:
        return item in self._workfile.series
    def __getitem__(self, index):
        return list(self._workfile.series)[index]
    def __call__(self) -> list[str]:
        return list(self._workfile.series)
    def __repr__(self) -> str:
        return repr(list(self._workfile.series))


class _NamesDescriptor:
    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return _CallableNamesView(instance)


Workfile.names = _NamesDescriptor()
EquationResult.name = property(
    lambda self: self.title.removeprefix("Equation: ").strip()
    if isinstance(self.title, str) else self.title
)


_original_equation_forecast = EquationResult.forecast


def _compat_equation_forecast(self, *args, **kwargs):
    """Bridge statsmodels OLS prediction to the EViews-style forecast API."""
    if args:
        if len(args) == 1 and isinstance(args[0], (int, __import__("numpy").integer)) and not any(k in kwargs for k in ("steps", "start", "end")):
            kwargs = dict(kwargs)
            kwargs["steps"] = int(args[0])
            args = ()
        else:
            return _original_equation_forecast(self, *args, **kwargs)

    structural = bool(kwargs.get("structural", False))
    process = self.error_process
    has_arma = bool(process.p or process.q or process.sar or process.sma)
    if structural or has_arma:
        return _original_equation_forecast(self, *args, **kwargs)

    model = self.result
    try:
        nobs = int(getattr(model, "nobs"))
    except Exception:
        return _original_equation_forecast(self, *args, **kwargs)

    steps = kwargs.get("steps")
    start = kwargs.get("start")
    end = kwargs.get("end")
    future_exog = kwargs.get("future_exog")
    if steps is not None and (start is not None or end is not None):
        return _original_equation_forecast(self, *args, **kwargs)
    if steps is not None:
        start_i, end_i = nobs, nobs + int(steps) - 1
    else:
        if start is None:
            return _original_equation_forecast(self, *args, **kwargs)
        start_i = int(start)
        end_i = int(end if end is not None else start)
    horizon = end_i - start_i + 1
    if horizon < 1:
        return _original_equation_forecast(self, *args, **kwargs)

    class _OLSForecastProxy:
        def __init__(self, wrapped):
            self._wrapped = wrapped
        def __getattr__(self, name):
            return getattr(self._wrapped, name)
        def get_prediction(self, start=None, end=None, dynamic=False, **prediction_kwargs):
            np = __import__("numpy")
            model_exog = np.asarray(getattr(getattr(self._wrapped, "model", None), "exog", np.empty((0, 0))), dtype=float)
            if "exog" in prediction_kwargs:
                exog = np.asarray(prediction_kwargs["exog"], dtype=float)
                if exog.ndim == 1:
                    exog = exog.reshape(1, -1)
            elif start is not None and int(start) < model_exog.shape[0]:
                stop = int(end) + 1 if end is not None else int(start) + 1
                exog = model_exog[int(start):stop]
            else:
                params = getattr(self._wrapped, "params", None)
                names = list(getattr(params, "index", []))
                if names == ["C"]:
                    exog = np.ones((int(end) - int(start) + 1, 1))
                else:
                    raise ValueError("future_exog is required for out-of-sample forecasts")
            if exog.shape[0] != int(end) - int(start) + 1:
                raise ValueError("forecast exog length must equal forecast horizon")
            return self._wrapped.get_prediction(exog=exog)

    original_result = self.result
    self.result = _OLSForecastProxy(model)
    try:
        return _original_equation_forecast(self, *args, **kwargs)
    finally:
        self.result = original_result


EquationResult.forecast = _compat_equation_forecast


__all__ = [
    "TimeSeries", "Workfile", "Expression", "ExpressionError", "evaluate", "Equation", "EquationResult", "UnifiedResult", "ResultTable",
    "ErrorProcess", "parse_error_terms",
    "ACFResult", "PACFResult", "acf", "pacf", "CorrelogramResult", "LjungBoxResult", "correlogram", "correlogram_ljung_box", "ljung_box", "format_correlogram_table", "format_correlogram", "interpret_correlogram",
    "DF_SPECIFICATIONS", "DF_CRITICAL_VALUES", "DF_F_CRITICAL_VALUES", "UnitRootResult", "SpecificationTestResult", "SequentialDFResult",
    "adf", "dickey_fuller", "kpss_test", "phillips_perron", "dickey_fuller_sequential", "classify_ts_ds", "difference", "trend_regression",
    "DecompositionResult", "moving_average", "weighted_moving_average", "exponential_smoothing", "holt", "holt_winters", "decompose", "seasonal_difference", "seasonal_indices", "fisher_seasonality_test",
    "TSResult", "fit_ar", "fit_ma", "fit_arma", "fit_arima", "fit_sarima", "estimate",
    "BoxJenkinsIdentificationResult", "identify_box_jenkins", "identify", "grid_search", "AutoARIMAResult", "autoarma",
    "EstimatedCandidate", "BoxJenkinsEstimationResult", "estimate_box_jenkins_candidates",
    "CandidateValidation", "BoxJenkinsValidationResult", "validate_box_jenkins_candidates",
    "BoxJenkinsSelectionResult", "select_box_jenkins_model",
    "BoxJenkinsForecastResult", "forecast_box_jenkins",
    "LinearStateSpace", "KalmanFilterResult", "kalman_filter", "local_level_filter", "KalmanSmootherResult", "kalman_smoother", "KalmanForecastResult", "kalman_forecast", "LocalLevelEstimateResult", "estimate_local_level",
    "KalmanInnovationDiagnosticsResult", "kalman_innovation_diagnostics", "StateSpaceAdequacyResult", "state_space_adequacy",
    "StateSpaceWorkflowResult", "run_local_level_workflow",
    "TestResult", "durbin_watson_test", "breusch_godfrey", "breusch_godfrey_raw", "box_pierce", "ljung_box", "jarque_bera", "mean_zero_test", "normality_ks", "variance_ratio_test", "breusch_pagan", "arch_test", "residual_correlogram", "residual_diagnostics_correlogram", "residual_diagnostics", "roots_report", "redundancy_check",
    "ForecastMetrics", "metrics", "prediction_interval", "restore_differences", "naive_forecast", "drift_forecast",
    "RegressionResult", "ols", "trend_terms", "CointegrationTestResult", "CointegratingRegressionResult", "ECMResult", "JohansenResult", "VECMResult", "cointreg", "engle_granger", "phillips_ouliaris", "johansen", "vecm", "ecm",
    "polynomial_roots", "is_stationary_ar", "is_invertible_ma", "process_mean", "impulse_response", "inverse_ar_coefficients", "theoretical_ar_acf", "theoretical_ma_acf",
    "plot_series", "plot_correlogram", "plot_eviews_correlogram", "plot_forecast", "plot_decomposition",
]

from .parity import ParityReport, Comparison, compare_number, compare_array, compare_dataframe, compare_equation_fixture, compare_forecast, compare_autoarma_reference, validate_forecast_reference_metadata, compare_johansen_reference, compare_breusch_godfrey_reference, validate_equations, assert_reports_pass, load_fixture, reports_dataframe, validate_fixture_schema
