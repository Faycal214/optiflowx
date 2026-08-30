"""Numerical validation utilities for StochX reference fixtures.

The parity layer deliberately validates *observable statistical behavior*.
It never imports, executes, or depends on EViews binaries or proprietary files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Iterable, Mapping
import math
import numpy as np
import pandas as pd


@dataclass
class Comparison:
    name: str
    passed: bool
    actual: float | str | None
    expected: float | str | None
    abs_error: float | None = None
    rel_error: float | None = None
    tolerance: float | None = None
    note: str = ""


@dataclass
class ParityReport:
    model: str
    comparisons: list[Comparison] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.comparisons)

    @property
    def failures(self) -> list[Comparison]:
        return [item for item in self.comparisons if not item.passed]

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "passed": self.passed,
            "checks": len(self.comparisons),
            "failures": len(self.failures),
        }

    def text(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"[{status}] {self.model}"]
        for item in self.comparisons:
            flag = "PASS" if item.passed else "FAIL"
            detail = ""
            if item.abs_error is not None:
                detail = f" abs={item.abs_error:.6g}"
            if item.note:
                detail += f" {item.note}"
            lines.append(f"  [{flag}] {item.name}{detail}")
        return "\n".join(lines)


def display_tolerance(expected_text: str) -> float:
    """Half a unit of the last displayed decimal place."""
    exponent = Decimal(str(expected_text)).as_tuple().exponent
    if exponent >= 0:
        return 0.5
    return float(Decimal("0.5") * (Decimal(10) ** exponent))


def compare_number(
    actual: float,
    expected: float | str,
    *,
    name: str,
    atol: float | None = None,
    rtol: float = 1e-8,
) -> Comparison:
    expected_value = float(expected)
    actual_value = float(actual)
    if atol is None and isinstance(expected, str):
        atol = display_tolerance(expected)
    elif atol is None:
        atol = 1e-8
    if not (math.isfinite(actual_value) and math.isfinite(expected_value)):
        passed = math.isnan(actual_value) and math.isnan(expected_value)
        return Comparison(name, passed, actual_value, expected_value, note="non-finite comparison")
    abs_error = abs(actual_value - expected_value)
    rel_error = abs_error / max(abs(expected_value), np.finfo(float).eps)
    passed = abs_error <= atol + rtol * abs(expected_value)
    return Comparison(name, passed, actual_value, expected_value, abs_error, rel_error, atol)


def compare_text(
    actual: str,
    expected: str,
    *,
    name: str,
    normalize_whitespace: bool = False,
) -> Comparison:
    lhs, rhs = actual, expected
    if normalize_whitespace:
        lhs = " ".join(lhs.split())
        rhs = " ".join(rhs.split())
    return Comparison(name, lhs == rhs, lhs, rhs)


def compare_mapping(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    report: ParityReport,
    prefix: str = "",
    rtol: float = 1e-8,
) -> None:
    for key, expected_value in expected.items():
        name = f"{prefix}{key}"
        if key not in actual:
            report.comparisons.append(Comparison(name, False, None, expected_value, note="missing field"))
            continue
        value = actual[key]
        if isinstance(expected_value, Mapping):
            if not isinstance(value, Mapping):
                report.comparisons.append(Comparison(name, False, value, expected_value, note="expected mapping"))
            else:
                compare_mapping(value, expected_value, report=report, prefix=f"{name}.", rtol=rtol)
        elif isinstance(expected_value, str):
            try:
                report.comparisons.append(compare_number(float(value), expected_value, name=name, rtol=rtol))
            except (TypeError, ValueError):
                report.comparisons.append(compare_text(str(value), expected_value, name=name))
        elif isinstance(expected_value, (int, float)):
            report.comparisons.append(compare_number(float(value), float(expected_value), name=name, rtol=rtol))
        else:
            report.comparisons.append(Comparison(name, value == expected_value, value, expected_value))


def compare_equation_fixture(
    result: Any,
    reference: Mapping[str, Any],
    *,
    model_name: str,
    rtol: float = 2e-8,
    root_rtol: float = 2e-2,
) -> ParityReport:
    """Compare an EquationResult/TSResult to an EViews reference fixture."""
    report = ParityReport(model_name)

    expected_nobs = reference.get("nobs")
    if expected_nobs is not None:
        report.comparisons.append(
            Comparison("nobs", int(result.nobs) == int(expected_nobs), result.nobs, expected_nobs)
        )

    expected_orders = reference.get("orders", {})
    if expected_orders:
        actual_process = getattr(result, "error_process", None)
        for side in ("ar", "ma", "sar", "sma"):
            if side not in expected_orders:
                continue
            actual = tuple(getattr(actual_process, side, ())) if actual_process is not None else ()
            expected = tuple(expected_orders[side])
            report.comparisons.append(
                Comparison(f"orders.{side}", actual == expected, actual, expected)
            )

    params = getattr(result, "params", None)
    if params is not None:
        params = params.to_dict()
        for term, fields in reference.get("coefficients", {}).items():
            if term not in params:
                report.comparisons.append(Comparison(f"coefficient.{term}", False, None, fields.get("coefficient"), note="missing coefficient"))
                continue
            report.comparisons.append(compare_number(params[term], fields["coefficient"], name=f"coefficient.{term}", rtol=rtol))

        for term, fields in reference.get("coefficients", {}).items():
            for source_key, result_attr in (
                ("std_error", "bse"),
                ("t_stat", "tvalues"),
                ("p_value", "pvalues"),
            ):
                values = getattr(result, result_attr, {})
                value = values.get(term) if hasattr(values, "get") else None
                if value is None:
                    report.comparisons.append(Comparison(f"{result_attr}.{term}", False, None, fields[source_key], note="missing statistic"))
                else:
                    report.comparisons.append(compare_number(value, fields[source_key], name=f"{result_attr}.{term}", rtol=rtol))

    stats = getattr(result, "statistics", lambda: {})()
    for label, expected in reference.get("statistics", {}).items():
        if label not in stats:
            report.comparisons.append(Comparison(f"statistics.{label}", False, None, expected, note="missing statistic"))
        else:
            report.comparisons.append(compare_number(stats[label], expected, name=f"statistics.{label}", rtol=rtol))

    roots = getattr(result, "roots", lambda: {})()
    expected_roots = reference.get("roots", {})
    for side, values in (("ar", expected_roots.get("ar", [])), ("ma", expected_roots.get("ma", []))):
        actual_values = np.asarray(roots.get(f"{side.upper()} roots", []), dtype=complex)
        if actual_values.size != len(values):
            report.comparisons.append(Comparison(f"roots.{side}.count", False, actual_values.size, len(values)))
            continue
        for idx, expected_text in enumerate(values):
            token = str(expected_text).replace("i", "")
            if "+" in token[1:]:
                real_text, imag_text = token.rsplit("+", 1)
                expected_complex = complex(float(real_text), float(imag_text))
            elif "-" in token[1:]:
                split_at = token[1:].find("-") + 1
                expected_complex = complex(float(token[:split_at]), float(token[split_at:]))
            else:
                expected_complex = complex(float(token), 0.0)
            actual_value = actual_values[idx]
            delta = abs(actual_value - expected_complex)
            tol = root_rtol * max(abs(expected_complex), 1.0)
            report.comparisons.append(
                Comparison(
                    f"roots.{side}[{idx}]",
                    bool(delta <= tol),
                    complex(actual_value),
                    expected_complex,
                    abs_error=float(delta),
                    tolerance=float(tol),
                )
            )
    return report


def compare_array(
    actual: Iterable[float],
    expected: Iterable[float | str],
    *,
    report: ParityReport,
    name: str,
    atol: float = 1e-8,
    rtol: float = 1e-8,
) -> None:
    lhs = np.asarray(list(actual), dtype=float).reshape(-1)
    rhs = np.asarray([float(v) for v in expected], dtype=float).reshape(-1)
    if lhs.size != rhs.size:
        report.comparisons.append(Comparison(name, False, lhs.size, rhs.size, note="length mismatch"))
        return
    for i, (a, e) in enumerate(zip(lhs, rhs)):
        report.comparisons.append(compare_number(a, e, name=f"{name}[{i}]", atol=atol, rtol=rtol))


def compare_forecast(
    actual,
    expected,
    *,
    model_name: str = "forecast",
    columns: tuple[str, ...] = ("Forecast", "Std. Error", "Lower", "Upper"),
    atol: float = 1e-8,
    rtol: float = 1e-8,
) -> ParityReport:
    """Compare forecast DataFrame columns against a captured reference."""
    report = ParityReport(model_name)
    for column in columns:
        if column not in actual or column not in expected:
            report.comparisons.append(
                Comparison(column, False, None if column not in actual else "present",
                           None if column not in expected else "present", note="missing column")
            )
            continue
        compare_array(
            actual[column].to_numpy(dtype=float),
            expected[column],
            report=report,
            name=column,
            atol=atol,
            rtol=rtol,
        )
    return report


def compare_dataframe(
    actual,
    expected,
    *,
    model_name: str,
    atol: float = 1e-8,
    rtol: float = 1e-8,
) -> ParityReport:
    """Compare two numeric DataFrames after requiring identical shape/labels."""
    report = ParityReport(model_name)
    if list(actual.columns) != list(expected.columns):
        report.comparisons.append(Comparison("columns", False, list(actual.columns), list(expected.columns)))
        return report
    if list(actual.index) != list(expected.index):
        report.comparisons.append(Comparison("index", False, list(actual.index), list(expected.index)))
        return report
    if actual.shape != expected.shape:
        report.comparisons.append(Comparison("shape", False, actual.shape, expected.shape))
        return report
    for row in actual.index:
        for column in actual.columns:
            a, e = actual.loc[row, column], expected.loc[row, column]
            if pd.isna(a) and pd.isna(e):
                continue
            try:
                report.comparisons.append(
                    compare_number(a, e, name=f"{row}.{column}", atol=atol, rtol=rtol)
                )
            except (TypeError, ValueError):
                report.comparisons.append(compare_text(str(a), str(e), name=f"{row}.{column}"))
    return report


def validate_equations(
    results: Mapping[str, Any],
    references: Mapping[str, Mapping[str, Any]],
    *,
    rtol: float = 2e-8,
    root_rtol: float = 2e-2,
) -> dict[str, ParityReport]:
    """Validate multiple named equation results against fixture references."""
    reports = {}
    for name, reference in references.items():
        if name not in results:
            report = ParityReport(name)
            report.comparisons.append(Comparison("equation", False, None, reference.get("specification"), note="missing result"))
            reports[name] = report
            continue
        reports[name] = compare_equation_fixture(
            results[name],
            reference,
            model_name=name,
            rtol=rtol,
            root_rtol=root_rtol,
        )
    return reports


def assert_reports_pass(reports: Mapping[str, ParityReport]) -> None:
    """Raise one compact assertion error containing every parity failure."""
    failures = [report.text() for report in reports.values() if not report.passed]
    if failures:
        raise AssertionError("\n\n".join(failures))
