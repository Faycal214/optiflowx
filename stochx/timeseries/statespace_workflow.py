"""Stage 11.7 end-to-end state-space workflow integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .statespace import KalmanFilterResult
from .statespace_adequacy import StateSpaceAdequacyResult, state_space_adequacy
from .statespace_diagnostics import (
    KalmanInnovationDiagnosticsResult,
    kalman_innovation_diagnostics,
)
from .statespace_estimation import LocalLevelEstimateResult, estimate_local_level
from .statespace_forecasting import KalmanForecastResult, kalman_forecast
from .statespace_smoothing import KalmanSmootherResult, kalman_smoother


@dataclass(frozen=True)
class StateSpaceWorkflowResult:
    """Auditable result for the complete Stage 11 local-level workflow."""

    estimation: LocalLevelEstimateResult
    filter_result: KalmanFilterResult
    smoother: KalmanSmootherResult
    innovation_diagnostics: KalmanInnovationDiagnosticsResult
    adequacy: StateSpaceAdequacyResult
    forecast: KalmanForecastResult | None

    def __post_init__(self) -> None:
        if not isinstance(self.estimation, LocalLevelEstimateResult):
            raise TypeError("estimation must be a LocalLevelEstimateResult")
        if not isinstance(self.filter_result, KalmanFilterResult):
            raise TypeError("filter_result must be a KalmanFilterResult")
        if not isinstance(self.smoother, KalmanSmootherResult):
            raise TypeError("smoother must be a KalmanSmootherResult")
        if not isinstance(self.innovation_diagnostics, KalmanInnovationDiagnosticsResult):
            raise TypeError("innovation_diagnostics must be a KalmanInnovationDiagnosticsResult")
        if not isinstance(self.adequacy, StateSpaceAdequacyResult):
            raise TypeError("adequacy must be a StateSpaceAdequacyResult")
        if self.forecast is not None and not isinstance(self.forecast, KalmanForecastResult):
            raise TypeError("forecast must be a KalmanForecastResult or None")
        if self.smoother.filter_result is not self.filter_result:
            raise ValueError("smoother must reuse the workflow filter_result")

    @property
    def adequate(self) -> bool:
        """Whether the configured innovation adequacy checks pass."""
        return self.adequacy.adequate

    @property
    def success(self) -> bool:
        """Whether estimation succeeded and the workflow produced a result."""
        return bool(self.estimation.success)



def run_local_level_workflow(
    observations: np.ndarray | Iterable[float],
    *,
    initial_level: float | None = None,
    initial_variance: float | None = None,
    start: tuple[float, float] = (0.1, 0.9),
    diagnostic_lags: int = 12,
    alpha: float = 0.05,
    forecast_steps: int = 0,
) -> StateSpaceWorkflowResult:
    """Run estimation, filtering, smoothing, diagnostics and forecasting.

    The function is a composition layer only: every numerical operation is
    delegated to the frozen Stage 10 and Stage 11 public APIs. Forecasting is
    optional and is skipped when ``forecast_steps`` is zero.
    """
    if not isinstance(diagnostic_lags, int) or isinstance(diagnostic_lags, bool) or diagnostic_lags < 1:
        raise ValueError("diagnostic_lags must be a positive integer")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if not isinstance(forecast_steps, int) or isinstance(forecast_steps, bool) or forecast_steps < 0:
        raise ValueError("forecast_steps must be a non-negative integer")

    estimation = estimate_local_level(
        observations,
        initial_level=initial_level,
        initial_variance=initial_variance,
        start=start,
    )
    if not estimation.success:
        raise ValueError(f"local-level estimation failed: {estimation.message}")

    filter_result = estimation.model
    # The fitted model is immediately filtered through the original observations;
    # the filtering result is the canonical input for all downstream stages.
    from .statespace import kalman_filter

    filtered = kalman_filter(observations, filter_result)
    smoother = kalman_smoother(observations, filter_result, filter_result=filtered)
    innovation_diagnostics = kalman_innovation_diagnostics(filtered)
    adequacy = state_space_adequacy(
        innovation_diagnostics,
        lags=diagnostic_lags,
        alpha=float(alpha),
    )
    forecast = None
    if forecast_steps:
        forecast = kalman_forecast(
            observations,
            filter_result,
            steps=forecast_steps,
            alpha=float(alpha),
            filter_result=filtered,
        )

    return StateSpaceWorkflowResult(
        estimation=estimation,
        filter_result=filtered,
        smoother=smoother,
        innovation_diagnostics=innovation_diagnostics,
        adequacy=adequacy,
        forecast=forecast,
    )
