"""Small, safe EViews-inspired expression engine for Workfiles."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .series import TimeSeries


class ExpressionError(ValueError):
    """Raised when a StochX time-series expression cannot be evaluated."""


@dataclass(frozen=True)
class Expression:
    """Parsed expression with its original source text."""

    source: str

    def evaluate(self, workfile) -> TimeSeries | float:
        """Evaluate the expression against a StochX Workfile."""
        return evaluate(self.source, workfile)


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _series(workfile, name: str) -> TimeSeries:
    try:
        return workfile.get(name)
    except KeyError as exc:
        raise ExpressionError(f"unknown series {name!r}") from exc


def _constant_like(series: TimeSeries, value: float, *, name: str = "C") -> TimeSeries:
    return TimeSeries(np.full(series.nobs, float(value)), index=series.index, name=name, frequency=series.frequency)


def _binary(left: Any, right: Any, op) -> TimeSeries | float:
    if isinstance(left, TimeSeries) and isinstance(right, TimeSeries):
        if left.nobs != right.nobs:
            raise ExpressionError("series must have the same number of observations")
        values = op(left.values, right.values)
        return TimeSeries(values, index=left.index, name="expression", frequency=left.frequency)
    if isinstance(left, TimeSeries):
        return TimeSeries(op(left.values, right), index=left.index, name="expression", frequency=left.frequency)
    if isinstance(right, TimeSeries):
        return TimeSeries(op(left, right.values), index=right.index, name="expression", frequency=right.frequency)
    return float(op(left, right))


def _unary(value: Any, op) -> TimeSeries | float:
    if isinstance(value, TimeSeries):
        return TimeSeries(op(value.values), index=value.index, name="expression", frequency=value.frequency)
    return float(op(value))


def _lag(series: TimeSeries, periods: int) -> TimeSeries:
    return series.lag(periods)


def _function(name: str, args: list[Any], workfile) -> TimeSeries | float:
    upper = name.upper()
    if upper == "D":
        if not args or not isinstance(args[0], TimeSeries):
            raise ExpressionError("D() expects a series")
        order = int(args[1]) if len(args) > 1 else 1
        return args[0].diff(order)
    if upper == "DLOG":
        if not args or not isinstance(args[0], TimeSeries):
            raise ExpressionError("DLOG() expects a series")
        transformed = args[0].log()
        order = int(args[1]) if len(args) > 1 else 1
        return transformed.diff(order)
    if upper == "LOG":
        if len(args) != 1 or not isinstance(args[0], TimeSeries):
            raise ExpressionError("LOG() expects one series")
        return args[0].log()
    if upper in {
        "MEAN", "@MEAN", "MEDIAN", "@MEDIAN", "MAX", "@MAX", "MIN", "@MIN",
        "STDEV", "@STDEV", "STDEVP", "@STDEVP", "VAR", "@VAR", "VARP", "@VARP",
        "OBS", "@OBS",
    }:
        if len(args) != 1 or not isinstance(args[0], TimeSeries):
            raise ExpressionError(f"@{upper.lower()}() expects one series")
        values = args[0].values
        if getattr(workfile, "sample_mask", None) is not None:
            values = values[workfile.sample_indexer]
        values = values[np.isfinite(values)]
        if values.size == 0:
            return float("nan")
        if upper in {"MEAN", "@MEAN"}:
            return float(np.mean(values))
        if upper in {"MEDIAN", "@MEDIAN"}:
            return float(np.median(values))
        if upper in {"MAX", "@MAX"}:
            return float(np.max(values))
        if upper in {"MIN", "@MIN"}:
            return float(np.min(values))
        if upper in {"STDEV", "@STDEV"}:
            return float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
        if upper in {"STDEVP", "@STDEVP"}:
            return float(np.std(values, ddof=0))
        if upper in {"VAR", "@VAR", "VARP", "@VARP"}:
            return float(np.var(values, ddof=0))
        return float(values.size)
    raise ExpressionError(f"unsupported function {name!r}")


def _eval_node(node: ast.AST, workfile) -> TimeSeries | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        upper_name = node.id.upper()
        if upper_name == "C":
            if not workfile.series:
                raise ExpressionError("C requires at least one series in the workfile")
            base = next(iter(workfile.series.values()))
            return _constant_like(base, 1.0)
        if upper_name == "TREND":
            if not workfile.series:
                raise ExpressionError("@TREND requires at least one series in the workfile")
            base = next(iter(workfile.series.values()))
            return TimeSeries(np.arange(base.nobs, dtype=float), index=base.index, name="@TREND", frequency=base.frequency)
        return _series(workfile, node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _eval_node(node.operand, workfile)
        return _unary(operand, lambda x: +x if isinstance(node.op, ast.UAdd) else -x)
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, workfile)
        result = None
        current = left
        for operator, comparator in zip(node.ops, node.comparators):
            right = _eval_node(comparator, workfile)
            if isinstance(operator, ast.Gt):
                op = lambda a, b: a > b
            elif isinstance(operator, ast.GtE):
                op = lambda a, b: a >= b
            elif isinstance(operator, ast.Lt):
                op = lambda a, b: a < b
            elif isinstance(operator, ast.LtE):
                op = lambda a, b: a <= b
            elif isinstance(operator, ast.Eq):
                op = lambda a, b: a == b
            elif isinstance(operator, ast.NotEq):
                op = lambda a, b: a != b
            else:
                raise ExpressionError("unsupported comparison operator")
            comparison = _binary(current, right, op)
            if result is None:
                result = comparison
            else:
                result = _binary(result, comparison, lambda a, b: np.logical_and(a, b))
            current = right
        if result is None:
            raise ExpressionError("comparison requires at least one operator")
        if isinstance(result, TimeSeries):
            return TimeSeries(result.values.astype(float), index=result.index, name="expression", frequency=result.frequency)
        return float(bool(result))

    if isinstance(node, ast.BoolOp):
        if len(node.values) < 2:
            raise ExpressionError("logical expressions require at least two operands")
        values = [_eval_node(value, workfile) for value in node.values]
        op = np.logical_and if isinstance(node.op, ast.And) else np.logical_or if isinstance(node.op, ast.Or) else None
        if op is None:
            raise ExpressionError("unsupported logical operator")
        result = values[0]
        for value in values[1:]:
            result = _binary(result, value, op)
        if isinstance(result, TimeSeries):
            return TimeSeries(result.values.astype(float), index=result.index, name="expression", frequency=result.frequency)
        return float(bool(result))

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = _eval_node(node.operand, workfile)
        if isinstance(value, TimeSeries):
            return TimeSeries((~value.values.astype(bool)).astype(float), index=value.index, name="expression", frequency=value.frequency)
        return float(not bool(value))

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, workfile)
        right = _eval_node(node.right, workfile)
        operators = {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.Div: lambda a, b: a / b,
            ast.Pow: lambda a, b: a**b,
        }
        for klass, op in operators.items():
            if isinstance(node.op, klass):
                return _binary(left, right, op)
        raise ExpressionError("unsupported operator")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExpressionError("only simple named functions are allowed")
        function_name = node.func.id
        if _IDENTIFIER.match(function_name) is None:
            raise ExpressionError("invalid function name")
        args = [_eval_node(arg, workfile) for arg in node.args]
        # EViews-style X(-1), X(1) syntax uses negative integers for lags and
        # positive integers for leads. TimeSeries.lag() uses the opposite sign
        # convention internally, so translate the expression sign here.
        if function_name in workfile.series and len(args) == 1 and isinstance(args[0], (int, float)):
            periods = int(args[0])
            if periods != args[0]:
                raise ExpressionError("lag/lead periods must be integers")
            return _lag(_series(workfile, function_name), -periods)
        return _function(function_name, args, workfile)
    raise ExpressionError(f"unsupported expression component: {ast.dump(node, include_attributes=False)}")


def evaluate(source: str, workfile) -> TimeSeries | float:
    """Evaluate an EViews-inspired expression against a Workfile.

    Supported examples include ``GDP``, ``GDP(-1)``, ``GDP(1)``, ``D(GDP)``,
    ``DLOG(GDP)``, ``LOG(GDP)``, ``@TREND``, ``GDP(-1) + 0.5*CONS`` and basic statistics
    such as ``@mean(GDP)``.
    """
    if not isinstance(source, str) or not source.strip():
        raise ExpressionError("expression must be a non-empty string")
    text = source.strip()
    # Python's AST cannot parse EViews identifiers beginning with "@".
    text = re.sub(r"@TREND\b", "TREND", text, flags=re.IGNORECASE)
    text = re.sub(r"@(?=mean|median|max|min|var|stdev|stdevp|varp|obs)\b", "", text, flags=re.IGNORECASE)
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"invalid expression {source!r}") from exc
    result = _eval_node(tree.body, workfile)
    if isinstance(result, TimeSeries):
        return result.copy(name=source.strip())
    return float(result)
