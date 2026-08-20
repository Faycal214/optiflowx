import numpy as np
import pandas as pd
import pytest

from stochx.timeseries import (
    ar,
    arma,
    estimate_box_jenkins_candidates,
    forecast_box_jenkins,
    select_box_jenkins_model,
)
from stochx.timeseries.box_jenkins_selection import BoxJenkinsSelectionResult


def _selected(y, order=(1, 0, 0)):
    estimation = estimate_box_jenkins_candidates(y, (order,))
    candidate = estimation.candidates[0]
    from stochx.timeseries.box_jenkins_validation import BoxJenkinsValidationResult, CandidateValidation

    validation = BoxJenkinsValidationResult((CandidateValidation(
        order=order,
        estimation_success=True,
        serially_adequate=True,
        adequate=True,
        residual_nobs=candidate.estimation_nobs,
        validation_lags=1,
        alpha=0.05,
        residual_correlogram=None,
        mean_test=None,
        normality_test=None,
        ks_test=None,
        arch_test=None,
        required_lag_pvalues=(0.9,),
        failed_checks=(),
        rationale="synthetic eligible candidate",
        estimated_candidate=candidate,
    ),))
    return select_box_jenkins_model(validation, criterion="aic")


def test_stage9_6_forecast_has_horizon_intervals_and_metadata():
    selection = _selected(ar(1, [0.55], 220, rng=7))
    result = forecast_box_jenkins(selection, steps=5, alpha=0.10)

    assert result.order == (1, 0, 0)
    assert result.horizon == 5
    assert result.alpha == 0.10
    assert result.confidence_level == 0.90
    assert len(result.index) == 5
    assert result.table().shape == (5, 4)
    assert np.isfinite(result.forecast).all()
    assert np.isfinite(result.lower).all()
    assert np.isfinite(result.upper).all()
    assert np.all(result.lower <= result.forecast)
    assert np.all(result.forecast <= result.upper)
    assert not result.forecast.flags.writeable
    assert result.metadata()["criterion"] == "aic"


def test_stage9_6_uses_datetime_index_when_model_has_datetime_index():
    dates = pd.date_range("2020-01-01", periods=80, freq="D")
    series = pd.Series(ar(1, [0.4], 80, rng=2).values, index=dates)
    selection = _selected(series)
    result = forecast_box_jenkins(selection, steps=3)

    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index[0] == dates[-1] + pd.Timedelta(days=1)
    assert result.index[-1] == dates[-1] + pd.Timedelta(days=3)


def test_stage9_6_explicit_forecast_index_must_match_horizon():
    selection = _selected(arma(p=1, q=1, phi=[0.3], theta=[0.2], n=120, rng=4), order=(1, 0, 1))
    with pytest.raises(ValueError, match="forecast_index length"):
        forecast_box_jenkins(selection, steps=3, forecast_index=[1, 2])


def test_stage9_6_requires_selected_adequate_model():
    validation = __import__("stochx.timeseries", fromlist=["BoxJenkinsValidationResult", "CandidateValidation"]).BoxJenkinsValidationResult(())
    selection = select_box_jenkins_model(validation)
    with pytest.raises(ValueError, match="selected adequate model"):
        forecast_box_jenkins(selection, steps=2)


def test_stage9_6_restores_explicit_differenced_scale():
    selection = _selected(ar(1, [0.5], 180, rng=9))
    result = forecast_box_jenkins(
        selection,
        steps=3,
        forecast_on_differenced_scale=True,
        last_levels=[10.0, 11.0, 12.0],
        restoration_order=1,
    )

    assert result.restored_from_differences is True
    assert result.scale == "original"
    assert result.restoration_order == 1
    assert result.restoration_history_nobs == 3
    assert np.all(result.lower <= result.forecast)
    assert np.all(result.forecast <= result.upper)


def test_stage9_6_differenced_restore_requires_history():
    selection = _selected(ar(1, [0.5], 100, rng=5))
    with pytest.raises(ValueError, match="last_levels"):
        forecast_box_jenkins(selection, steps=2, forecast_on_differenced_scale=True)
