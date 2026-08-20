import numpy as np
import pandas as pd

from stochx.timeseries import (
    correlogram,
    estimate_box_jenkins_candidates,
    forecast_box_jenkins,
    select_box_jenkins_model,
    validate_box_jenkins_candidates,
)
from tests.fixtures.stage9_box_jenkins_fixture import EXPECTED, SETTINGS, make_fixture_series


RTOL = 5e-6
ATOL = 5e-7


def test_stage9_7_end_to_end_estimation_validation_selection_and_forecast():
    y = make_fixture_series()
    orders = ((0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1))

    estimation = estimate_box_jenkins_candidates(y, orders)
    assert estimation.orders == orders
    assert all(candidate.success for candidate in estimation.candidates)
    assert all(candidate.converged for candidate in estimation.successful)

    validation = validate_box_jenkins_candidates(
        estimation,
        lags=SETTINGS["validation_lags"],
        alpha=SETTINGS["alpha"],
    )
    assert validation.has_adequate_model
    assert validation.candidates[0].eligible is False
    assert validation.candidates[2].eligible is True
    assert validation.candidates[3].eligible is True

    selection = select_box_jenkins_model(
        validation,
        criterion=SETTINGS["criterion"],
        tie_tolerance=1e-8,
    )
    assert selection.status == "selected"
    assert selection.selected_order == EXPECTED["selected_order"]
    assert selection.rationale

    selected = selection.selected
    assert selected is not None
    np.testing.assert_allclose(selected.params, EXPECTED["params"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(selected.standard_errors, EXPECTED["bse"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(selected.tvalues, EXPECTED["tvalues"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(selected.pvalues, EXPECTED["pvalues"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(selected.ar_roots, EXPECTED["ar_roots"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(selected.ma_roots, EXPECTED["ma_roots"], rtol=RTOL, atol=ATOL)
    assert abs(selected.log_likelihood - EXPECTED["llf"]) <= ATOL
    assert abs(selected.sigma_sq - EXPECTED["sigma_sq"]) <= ATOL
    assert abs(selected.aic - EXPECTED["aic"]) <= ATOL
    assert abs(selected.bic - EXPECTED["sc"]) <= ATOL
    assert abs(selected.hq - EXPECTED["hq"]) <= ATOL

    forecast = forecast_box_jenkins(
        selection,
        steps=SETTINGS["forecast_steps"],
        alpha=SETTINGS["alpha"],
    )
    np.testing.assert_allclose(forecast.forecast, EXPECTED["forecast"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(forecast.standard_error, EXPECTED["forecast_se"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(forecast.lower, EXPECTED["lower"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(forecast.upper, EXPECTED["upper"], rtol=RTOL, atol=ATOL)
    assert forecast.metadata()["order"] == EXPECTED["selected_order"]
    assert forecast.metadata()["criterion"] == SETTINGS["criterion"]
    assert forecast.metadata()["forecast_horizon"] == SETTINGS["forecast_steps"]


def test_stage9_7_datetime_index_and_prediction_interval_invariants():
    dates = pd.date_range("2000-01-01", periods=SETTINGS["nobs"], freq="D")
    series = pd.Series(make_fixture_series(), index=dates)
    estimation = estimate_box_jenkins_candidates(series, ((1, 0, 1),))
    validation = validate_box_jenkins_candidates(estimation, lags=SETTINGS["validation_lags"])
    selection = select_box_jenkins_model(validation)

    forecast = forecast_box_jenkins(selection, steps=5, alpha=0.10)
    assert isinstance(forecast.index, pd.DatetimeIndex)
    assert forecast.index[0] == dates[-1] + pd.Timedelta(days=1)
    assert forecast.index[-1] == dates[-1] + pd.Timedelta(days=5)
    assert np.all(forecast.lower <= forecast.forecast)
    assert np.all(forecast.forecast <= forecast.upper)
    assert forecast.confidence_level == 0.90


def test_stage9_7_transformation_restoration_and_metadata():
    estimation = estimate_box_jenkins_candidates(make_fixture_series(), ((1, 0, 1),))
    validation = validate_box_jenkins_candidates(estimation, lags=SETTINGS["validation_lags"])
    selection = select_box_jenkins_model(validation)

    forecast = forecast_box_jenkins(
        selection,
        steps=3,
        forecast_on_differenced_scale=True,
        last_levels=[10.0, 11.0, 12.0],
        restoration_order=1,
    )
    assert forecast.restored_from_differences is True
    assert forecast.scale == "original"
    assert forecast.restoration_order == 1
    assert forecast.restoration_history_nobs == 3
    assert forecast.metadata()["restored_from_differences"] is True


def test_stage9_7_audits_frozen_stage8_correlogram_contract():
    corr = correlogram(make_fixture_series(), nlags=8, model_df=2, alpha=SETTINGS["alpha"])
    assert corr.nobs == SETTINGS["nobs"]
    assert corr.nlags == 8
    assert corr.model_df == 2
    assert corr.DF.tolist() == list(range(-1, 7))
    assert corr.table().shape == (8, 10)
    assert list(corr.table().columns) == [
        "Lag", "AC", "PAC", "Q-Stat", "Prob.", "DF",
        "AC Lower", "AC Upper", "PAC Lower", "PAC Upper",
    ]
    assert not corr.ac.flags.writeable
    assert not corr.pac.flags.writeable
