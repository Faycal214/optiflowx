"""ARMA-error estimation helpers for EViews-style equations."""

from __future__ import annotations

from dataclasses import dataclass
import re


_TERM_RE = re.compile(r"^(AR|MA|SAR|SMA)\((-?\d+)\)$", re.IGNORECASE)
_RANGE_RE = re.compile(r"^(AR|MA)\(\s*(-?\d+)\s+to\s+(-?\d+)\s*\)$", re.IGNORECASE)


@dataclass(frozen=True)
class ErrorProcess:
    """ARMA disturbance specification."""

    p: tuple[int, ...] = ()
    q: tuple[int, ...] = ()

    @property
    def max_p(self) -> int:
        return max(self.p, default=0)

    @property
    def max_q(self) -> int:
        return max(self.q, default=0)

    @property
    def order(self) -> tuple[int, int]:
        return self.max_p, self.max_q


def parse_error_terms(tokens: list[str]) -> tuple[list[str], ErrorProcess]:
    """Separate AR/MA error terms from observed regressors."""
    regressors: list[str] = []
    ar: set[int] = set()
    ma: set[int] = set()

    for token in tokens:
        match = _TERM_RE.match(token)
        if match:
            order = int(match.group(2))
            if order <= 0:
                raise ValueError("AR/MA error orders must be positive")
            (ar if match.group(1).upper() == "AR" else ma).add(order)
            continue
        match = _RANGE_RE.match(token)
        if match:
            kind = match.group(1).upper()
            start = int(match.group(2))
            end = int(match.group(3))
            if start <= 0 or end <= 0:
                raise ValueError("AR/MA error orders must be positive")
            step = 1 if end >= start else -1
            target = ar if kind == "AR" else ma
            target.update(range(start, end + step, step))
            continue
        regressors.append(token)

    return regressors, ErrorProcess(tuple(sorted(ar)), tuple(sorted(ma)))
