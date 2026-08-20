import numpy as np

from stochx.timeseries import (
    BoxJenkinsSelectionResult,
    CandidateValidation,
    EstimatedCandidate,
    BoxJenkinsValidationResult,
    select_box_jenkins_model,
)


def _candidate(order, *, aic, bic, hq, params=2, success=True):
    return EstimatedCandidate(
        order=order,
        success=success,
        model_name="ARMA",
        estimation_nobs=100 if success else None,
        params=np.zeros(params),
        standard_errors=np.ones(params),
        tvalues=np.ones(params),
        pvalues=np.full(params, 0.5),
        log_likelihood=-100.0,
        sigma_sq=1.0,
        aic=aic,
        bic=bic,
        hq=hq,
        ar_roots=np.asarray([], dtype=complex),
        ma_roots=np.asarray([], dtype=complex),
        converged=success,
        residuals=np.ones(100),
        error=None if success else "failed",
    )


def _validation(candidates):
    checks = tuple(
        CandidateValidation(
            order=c.order,
            estimation_success=c.success,
            serially_adequate=True if c.success else False,
            adequate=True if c.success else False,
            residual_nobs=100 if c.success else 0,
            validation_lags=8,
            alpha=0.05,
            residual_correlogram=None,
            mean_test=None,
            normality_test=None,
            ks_test=None,
            arch_test=None,
            required_lag_pvalues=(0.5,) if c.success else (),
            failed_checks=() if c.success else ("estimation",),
            rationale="eligible" if c.success else "failed",
            error=c.error,
            estimated_candidate=c,
        )
        for c in candidates
    )
    return BoxJenkinsValidationResult(checks)


def test_stage9_5_selects_lowest_explicit_criterion_only_among_eligible_candidates():
    candidates = (
        _candidate((1, 0, 0), aic=100, bic=105, hq=102),
        _candidate((1, 0, 1), aic=90, bic=110, hq=99),
        _candidate((2, 0, 0), aic=80, bic=101, hq=95, success=False),
    )
    result = select_box_jenkins_model(_validation(candidates), criterion="aic")

    assert isinstance(result, BoxJenkinsSelectionResult)
    assert result.status == "selected"
    assert result.selected_order == (1, 0, 1)
    assert result.eligible_orders == ((1, 0, 0), (1, 0, 1))
    assert "AIC" in result.rationale


def test_stage9_5_supports_sc_bic_and_hq_aliases():
    candidates = (
        _candidate((1, 0, 0), aic=90, bic=80, hq=85),
        _candidate((0, 0, 1), aic=80, bic=90, hq=70),
    )
    validation = _validation(candidates)

    assert select_box_jenkins_model(validation, criterion="sc").selected_order == (1, 0, 0)
    assert select_box_jenkins_model(validation, criterion="bic").selected_order == (1, 0, 0)
    assert select_box_jenkins_model(validation, criterion="hq").selected_order == (0, 0, 1)


def test_stage9_5_tie_uses_parameter_parsimony_then_order():
    candidates = (
        _candidate((2, 0, 0), aic=50.0, bic=50.0, hq=50.0, params=3),
        _candidate((1, 0, 1), aic=50.0 + 5e-9, bic=50.0, hq=50.0, params=2),
        _candidate((1, 0, 0), aic=50.0 + 5e-9, bic=50.0, hq=50.0, params=2),
    )
    result = select_box_jenkins_model(_validation(candidates), criterion="aic", tie_tolerance=1e-8)

    assert result.selected_order == (1, 0, 0)
    assert "tied within tolerance" in result.rationale
    assert "parsimony" in result.rationale


def test_stage9_5_no_adequate_model_is_explicit_and_not_forced():
    failed = _candidate((1, 0, 0), aic=10, bic=10, hq=10, success=False)
    result = select_box_jenkins_model(_validation((failed,)), criterion="aic")

    assert result.selected is None
    assert result.status == "no_adequate_model"
    assert result.has_selection is False
    assert "not performed" in result.rationale


def test_stage9_5_rejects_invalid_settings():
    candidates = (_candidate((1, 0, 0), aic=10, bic=10, hq=10),)
    validation = _validation(candidates)

    try:
        select_box_jenkins_model(validation, criterion="invalid")
    except ValueError as exc:
        assert "criterion" in str(exc)
    else:
        raise AssertionError("invalid criterion should raise ValueError")

    try:
        select_box_jenkins_model(validation, criterion="aic", tie_tolerance=-1)
    except ValueError as exc:
        assert "tie_tolerance" in str(exc)
    else:
        raise AssertionError("negative tie_tolerance should raise ValueError")


def test_stage9_5_ranking_table_is_deterministic():
    candidates = (
        _candidate((0, 0, 1), aic=20, bic=22, hq=21),
        _candidate((1, 0, 0), aic=10, bic=12, hq=11),
    )
    result = select_box_jenkins_model(_validation(candidates), criterion="aic")
    table = result.table()

    assert list(table.columns) == [
        "Rank", "p", "d", "q", "Parameters", "Criterion", "AIC", "SC", "HQ", "Selected"
    ]
    assert table[["p", "d", "q"]].to_records(index=False).tolist() == [(1, 0, 0), (0, 0, 1)]
    assert table["Selected"].tolist() == [True, False]
