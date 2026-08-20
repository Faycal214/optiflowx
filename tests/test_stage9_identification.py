import numpy as np
import pytest

import stochx.timeseries.identification as identification
from stochx.timeseries import BoxJenkinsIdentificationResult, identify_box_jenkins
from stochx.timeseries.stationarity import SpecificationTestResult, UnitRootResult


def test_stage9_explicit_differencing_and_candidate_generation_are_deterministic():
    y = np.cumsum(np.sin(np.arange(120) * 0.13) + 0.05)

    result = identify_box_jenkins(
        y,
        d=1,
        nlags=8,
        max_p=2,
        max_q=2,
    )

    assert isinstance(result, BoxJenkinsIdentificationResult)
    assert result.differencing_order == 1
    assert result.d == 1
    assert result.original.shape == (120,)
    assert result.transformed.shape == (119,)
    assert result.acf_result.nobs == 119
    assert result.pacf_result.nobs == 119
    assert all(order[1] == 1 for order in result.candidate_orders)
    assert result.candidate_orders == tuple(sorted(set(result.candidate_orders)))
    assert (0, 1, 0) in result.candidate_orders
    assert "no model is selected" in result.interpretation


def test_stage9_acf_pacf_hints_use_initial_contiguous_significant_runs(monkeypatch):
    class FakeResult:
        nlags = 6
        lags = np.arange(7)

        def significant(self):
            return np.array([False, True, True, False, True, False, False])

    class FakeACF(FakeResult):
        nobs = 80

    class FakePACF(FakeResult):
        nobs = 80

    monkeypatch.setattr(identification, "acf", lambda *args, **kwargs: FakeACF())
    monkeypatch.setattr(identification, "pacf", lambda *args, **kwargs: FakePACF())

    result = identify_box_jenkins(
        np.arange(80.0),
        d=0,
        nlags=6,
        max_p=2,
        max_q=2,
    )

    # Both ACF and PACF have contiguous significance through lag 2.
    assert result.ar_order_hint == 2
    assert result.ma_order_hint == 2
    assert result.acf_significant_lags == (1, 2, 4)
    assert result.pacf_significant_lags == (1, 2, 4)
    assert result.candidate_orders == tuple(sorted(result.candidate_orders))
    assert len(result.candidate_orders) == 9


def test_stage9_automatic_differencing_stops_at_first_stationary_level(monkeypatch):
    calls = []

    def fake_sequential(x, **kwargs):
        calls.append(np.asarray(x).size)
        reject = len(calls) == 2
        return type(
            "FakeSequential",
            (),
            {
                "selected": type("Selected", (), {"decision": "reject" if reject else "fail_to_reject", "rejects_null": reject})(),
                "nature": "stationary" if reject else "integrated",
                "common_lag": 1,
            },
        )()

    monkeypatch.setattr(identification, "dickey_fuller_sequential", fake_sequential)
    monkeypatch.setattr(identification, "acf", lambda x, **kwargs: identification._real_acf(x, **kwargs))
    monkeypatch.setattr(identification, "pacf", lambda x, **kwargs: identification._real_pacf(x, **kwargs))

    # Avoid replacing public functions permanently in the module under test.
    original_acf = identification.acf
    original_pacf = identification.pacf
    try:
        from stochx.timeseries.correlation import acf as real_acf, pacf as real_pacf
        identification._real_acf = real_acf
        identification._real_pacf = real_pacf
        result = identify_box_jenkins(np.cumsum(np.arange(80.0)), max_d=2, nlags=6)
    finally:
        del identification._real_acf
        del identification._real_pacf
        identification.acf = original_acf
        identification.pacf = original_pacf

    assert result.differencing_order == 1
    assert len(calls) == 2
    assert calls == [80, 79]
    assert result.stationarity_decisions[0].startswith("d=0:")
    assert result.stationarity_decisions[1].startswith("d=1:")


def test_stage9_invalid_identification_inputs_fail_explicitly():
    with pytest.raises(ValueError, match="at least two"):
        identify_box_jenkins([1.0], d=0)
    with pytest.raises(ValueError, match="infinite"):
        identify_box_jenkins([1.0, np.inf, 2.0], d=0)
    with pytest.raises(ValueError, match="d must be an integer"):
        identify_box_jenkins([1.0, 2.0, 3.0], d=3, max_d=2)
    with pytest.raises(ValueError, match="positive integer"):
        identify_box_jenkins([1.0, 2.0, 3.0], d=0, nlags=0)
    with pytest.raises(ValueError, match="non-negative integer"):
        identify_box_jenkins([1.0, 2.0, 3.0], d=0, max_p=-1)


def test_stage9_legacy_identify_api_remains_available():
    result = identification.identify(np.arange(50.0), nlags=6)
    assert {"ACF", "PACF", "ACF significant lags", "PACF significant lags"}.issubset(result)
