import json
from pathlib import Path

import numpy as np

from stochx.timeseries import LinearStateSpace, kalman_filter

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "stage10_statespace"


def _load(name):
    with (FIXTURE_DIR / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def _model(fixture):
    return LinearStateSpace(
        transition=np.asarray(fixture["transition"], dtype=float),
        design=np.asarray(fixture["design"], dtype=float),
        state_cov=np.asarray(fixture["state_cov"], dtype=float),
        observation_cov=np.asarray(fixture["observation_cov"], dtype=float),
        initial_state=np.asarray(fixture["initial_state"], dtype=float),
        initial_cov=np.asarray(fixture["initial_cov"], dtype=float),
    )


def _assert_fixture(result, fixture):
    expected = fixture["expected"]
    tol = fixture["tolerances"]
    np.testing.assert_allclose(
        result.predicted_state,
        expected["predicted_state"],
        rtol=tol["rtol"],
        atol=tol["atol"],
    )
    np.testing.assert_allclose(
        result.filtered_state,
        expected["filtered_state"],
        rtol=tol["rtol"],
        atol=tol["atol"],
    )
    np.testing.assert_allclose(
        result.predicted_cov,
        expected["predicted_cov"],
        rtol=tol["rtol"],
        atol=tol["atol"],
    )
    np.testing.assert_allclose(
        result.filtered_cov,
        expected["filtered_cov"],
        rtol=tol["rtol"],
        atol=tol["atol"],
    )
    if "innovations" in expected:
        np.testing.assert_allclose(
            result.innovations,
            expected["innovations"],
            rtol=tol["rtol"],
            atol=tol["atol"],
            equal_nan=True,
        )
    if "innovation_cov" in expected:
        np.testing.assert_allclose(
            result.innovation_cov,
            expected["innovation_cov"],
            rtol=tol["rtol"],
            atol=tol["atol"],
            equal_nan=True,
        )
    if "observed_dimensions" in expected:
        np.testing.assert_array_equal(result.observed_dimensions, expected["observed_dimensions"])
    assert result.nobs == expected["nobs"]
    assert result.effective_nobs == expected["effective_nobs"]
    assert result.missing_observations == expected["missing_observations"]
    assert np.isclose(
        result.loglik,
        expected["loglik"],
        rtol=tol["rtol"],
        atol=tol["loglik_atol"],
    )


def test_scalar_local_level_matches_hand_calculated_fixture():
    fixture = _load("scalar_local_level.json")
    result = kalman_filter(fixture["observations"], _model(fixture))
    _assert_fixture(result, fixture)


def test_scalar_dynamic_matches_independent_reference_fixture():
    fixture = _load("scalar_dynamic.json")
    result = kalman_filter(fixture["observations"], _model(fixture))
    _assert_fixture(result, fixture)


def test_multivariate_missing_matches_independent_reference_fixture():
    fixture = _load("multivariate_missing.json")
    result = kalman_filter(fixture["observations"], _model(fixture))
    _assert_fixture(result, fixture)


def test_fixture_inputs_are_not_mutated():
    fixture = _load("multivariate_missing.json")
    observations = np.asarray(fixture["observations"], dtype=float)
    model = _model(fixture)
    original_observations = observations.copy()
    matrices = [
        model.transition.copy(),
        model.design.copy(),
        model.state_cov.copy(),
        model.observation_cov.copy(),
        model.initial_state.copy(),
        model.initial_cov.copy(),
    ]

    kalman_filter(observations, model)

    np.testing.assert_array_equal(observations, original_observations, equal_nan=True)
    for current, original in zip(
        [
            model.transition,
            model.design,
            model.state_cov,
            model.observation_cov,
            model.initial_state,
            model.initial_cov,
        ],
        matrices,
    ):
        np.testing.assert_array_equal(current, original)


def test_fixture_covariances_are_symmetric():
    fixture = _load("multivariate_missing.json")
    result = kalman_filter(fixture["observations"], _model(fixture))
    np.testing.assert_allclose(
        result.filtered_cov,
        np.swapaxes(result.filtered_cov, -1, -2),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.predicted_cov,
        np.swapaxes(result.predicted_cov, -1, -2),
        rtol=1e-12,
        atol=1e-12,
    )


def test_fixture_runs_are_repeatable():
    fixture = _load("scalar_dynamic.json")
    model = _model(fixture)
    first = kalman_filter(fixture["observations"], model)
    second = kalman_filter(fixture["observations"], model)
    np.testing.assert_array_equal(first.filtered_state, second.filtered_state)
    np.testing.assert_array_equal(first.filtered_cov, second.filtered_cov)
    np.testing.assert_array_equal(first.innovations, second.innovations)
    assert first.loglik == second.loglik
