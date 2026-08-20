import numpy as np
import pytest

from stochx.timeseries import (
    BoxJenkinsValidationResult,
    ar,
    estimate_box_jenkins_candidates,
    validate_box_jenkins_candidates,
)


def test_stage9_4_validation_records_frozen_residual_correlogram_and_adequacy():
    y = ar(1, [0.0], 220, rng=3)
    estimation = estimate_box_jenkins_candidates(y, ((0, 0, 0), (1, 0, 0)))
    result = validate_box_jenkins_candidates(estimation, lags=8)

    assert isinstance(result, BoxJenkinsValidationResult)
    assert result.candidates[0].residual_correlogram is not None
    corr = result.candidates[0].residual_correlogram
    assert corr.model_df == 0
    assert corr.nlags == 8
    assert corr.DF.tolist() == list(range(1, 9))
    assert result.candidates[0].validation_lags == 8
    assert result.candidates[0].residual_nobs == 220
    assert isinstance(result.has_adequate_model, bool)


def test_stage9_4_serial_correlation_is_mandatory_and_rejects_correlated_residuals():
    y = ar(1, [0.92], 250, rng=7)
    estimation = estimate_box_jenkins_candidates(y, ((0, 0, 0),))
    result = validate_box_jenkins_candidates(estimation, lags=8)

    candidate = result.candidates[0]
    assert candidate.adequate is False
    assert candidate.serially_adequate is False
    assert "serial_correlation" in candidate.failed_checks
    assert candidate.eligible is False


def test_stage9_4_optional_checks_only_make_eligibility_stricter():
    y = ar(1, [0.0], 250, rng=11)
    estimation = estimate_box_jenkins_candidates(y, ((0, 0, 0),))

    base = validate_box_jenkins_candidates(estimation, lags=6)
    strict = validate_box_jenkins_candidates(
        estimation,
        lags=6,
        require_mean_zero=True,
        require_normality=True,
        require_arch_free=True,
    )

    assert strict.candidates[0].adequate <= base.candidates[0].adequate
    assert strict.candidates[0].residual_correlogram is not None
    assert strict.candidates[0].normality_test is not None
    assert strict.candidates[0].ks_test is not None
    assert strict.candidates[0].arch_test is not None


def test_stage9_4_failed_estimation_is_not_revalidated():
    from stochx.timeseries import box_jenkins_estimation as estimation_module

    real_estimate = estimation_module.estimate

    def fake_estimate(y, *, p, d, q):
        if (p, d, q) == (9, 0, 9):
            raise RuntimeError("synthetic estimation failure")
        return real_estimate(y, p=p, d=d, q=q)

    pytest.MonkeyPatch().setattr(estimation_module, "estimate", fake_estimate)
    y = ar(1, [0.35], 150, rng=4)
    estimation = estimation_module.estimate_box_jenkins_candidates(y, ((9, 0, 9),))
    result = validate_box_jenkins_candidates(estimation, lags=6)
    candidate = result.candidates[0]

    assert candidate.estimation_success is False
    assert candidate.residual_correlogram is None
    assert candidate.failed_checks == ("estimation",)
    assert candidate.adequate is False
    assert "synthetic estimation failure" in candidate.error


def test_stage9_4_no_adequate_candidate_is_explicit():
    y = ar(1, [0.99], 120, rng=8)
    estimation = estimate_box_jenkins_candidates(y, ((0, 0, 0),))
    result = validate_box_jenkins_candidates(estimation, lags=8)

    assert result.has_adequate_model is False
    assert result.adequate == ()
    assert len(result.inadequate) == 1


def test_stage9_4_rejects_invalid_validation_settings():
    y = ar(1, [0.2], 100, rng=1)
    estimation = estimate_box_jenkins_candidates(y, ((1, 0, 0),))

    with pytest.raises(ValueError, match="positive integer"):
        validate_box_jenkins_candidates(estimation, lags=0)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        validate_box_jenkins_candidates(estimation, alpha=1.0)
