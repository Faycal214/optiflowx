import numpy as np
import pytest

from stochx.timeseries import (
    BoxJenkinsEstimationResult,
    EstimatedCandidate,
    ar,
    estimate_box_jenkins_candidates,
)


def test_stage9_3_estimates_deterministic_candidate_sequence():
    y = ar(1, [0.55], 220, rng=7)
    orders = ((0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1))

    result = estimate_box_jenkins_candidates(y, orders)

    assert isinstance(result, BoxJenkinsEstimationResult)
    assert result.orders == orders
    assert all(candidate.success for candidate in result.candidates)
    assert all(candidate.converged for candidate in result.candidates)

    for candidate in result.successful:
        n_params = candidate.params.size
        assert candidate.standard_errors.size == n_params
        assert candidate.tvalues.size == n_params
        assert candidate.pvalues.size == n_params
        assert np.isfinite(candidate.params).all()
        assert np.isfinite(candidate.standard_errors).all()
        assert np.isfinite(candidate.tvalues).all()
        assert np.isfinite(candidate.pvalues).all()
        assert np.isfinite(candidate.log_likelihood)
        assert np.isfinite(candidate.sigma_sq)
        assert np.isfinite(candidate.aic)
        assert np.isfinite(candidate.bic)
        assert np.isfinite(candidate.hq)
        assert candidate.estimation_nobs is not None
        assert candidate.residuals.size > 0
        assert candidate.ts_result is not None
        assert len(candidate.coefficient_names) == n_params
        assert np.array_equal(candidate.coefficients, candidate.params)
        assert np.array_equal(candidate.se, candidate.standard_errors)
        assert np.array_equal(candidate.t_stats, candidate.tvalues)
        assert np.array_equal(candidate.p_values, candidate.pvalues)
        assert candidate.llf == candidate.log_likelihood
        assert candidate.SIGMASQ == candidate.sigma_sq
        assert candidate.SC == candidate.bic
        assert candidate.HQ == candidate.hq
        assert candidate.convergence is candidate.converged
        assert not candidate.params.flags.writeable
        assert not candidate.standard_errors.flags.writeable
        assert not candidate.residuals.flags.writeable


def test_stage9_3_table_is_stable_and_contains_required_model_statistics():
    y = ar(1, [0.4], 180, rng=11)
    result = estimate_box_jenkins_candidates(y, ((1, 0, 0), (1, 0, 1)))

    table = result.table()
    assert list(table.columns) == [
        "p", "d", "q", "success", "model", "nobs", "LogLik",
        "SIGMASQ", "AIC", "SC", "HQ", "converged", "error",
    ]
    assert table[["p", "d", "q"]].to_records(index=False).tolist() == [
        (1, 0, 0),
        (1, 0, 1),
    ]
    assert table["success"].tolist() == [True, True]
    assert table["converged"].tolist() == [True, True]
    assert np.isfinite(table[["LogLik", "SIGMASQ", "AIC", "SC", "HQ"]].to_numpy()).all()


def test_stage9_3_one_failed_candidate_does_not_abort_remaining_candidates(monkeypatch):
    import stochx.timeseries.box_jenkins_estimation as estimation

    real_estimate = estimation.estimate

    def fake_estimate(y, *, p, d, q):
        if (p, d, q) == (9, 0, 9):
            raise RuntimeError("synthetic estimation failure")
        return real_estimate(y, p=p, d=d, q=q)

    monkeypatch.setattr(estimation, "estimate", fake_estimate)

    y = ar(1, [0.35], 150, rng=4)
    result = estimate_box_jenkins_candidates(y, ((9, 0, 9), (1, 0, 0)))

    assert result.candidates[0].success is False
    assert result.candidates[0].converged is False
    assert "synthetic estimation failure" in result.candidates[0].error
    assert result.candidates[1].success is True
    assert result.candidates[1].ts_result is not None


def test_stage9_3_rejects_duplicate_or_invalid_candidate_orders():
    y = ar(1, [0.25], 80, rng=2)

    with pytest.raises(ValueError, match="must not contain duplicates"):
        estimate_box_jenkins_candidates(y, ((1, 0, 0), (1, 0, 0)))

    with pytest.raises(ValueError, match=r"non-negative \(p, d, q\) triples"):
        estimate_box_jenkins_candidates(y, ((1, -1, 0),))


def test_stage9_3_white_noise_candidate_uses_existing_arima_estimator():
    y = np.linspace(1.0, 10.0, 80) + np.random.default_rng(3).normal(0.0, 0.2, 80)
    result = estimate_box_jenkins_candidates(y, ((0, 0, 0),))
    candidate = result.candidates[0]
    assert candidate.success is True
    assert candidate.model_name == "ARIMA"
    assert candidate.ts_result is not None
    assert np.isfinite(candidate.sigma_sq)


def test_stage9_3_failed_candidate_snapshot_remains_immutable():
    candidate = EstimatedCandidate(
        order=(99, 0, 99),
        success=False,
        model_name="",
        estimation_nobs=None,
        params=np.array([]),
        standard_errors=np.array([]),
        tvalues=np.array([]),
        pvalues=np.array([]),
        log_likelihood=np.nan,
        sigma_sq=np.nan,
        aic=np.nan,
        bic=np.nan,
        hq=np.nan,
        ar_roots=np.array([], dtype=complex),
        ma_roots=np.array([], dtype=complex),
        converged=False,
        residuals=np.array([]),
        error="failed",
    )
    assert candidate.order == (99, 0, 99)
    assert candidate.success is False
    assert candidate.error == "failed"
