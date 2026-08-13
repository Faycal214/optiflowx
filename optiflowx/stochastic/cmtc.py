"""Homogeneous continuous-time Markov chains from MSPRO Chapter 3.

The public API mirrors the mathematical objects developed in the supplied
course chapter: the generator Q, transition matrices P(t), Kolmogorov
equations, holding times, the embedded jump chain, stationarity, recurrence,
and continuous-time occupation behavior.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Hashable

import numpy as np
from scipy.linalg import expm

from .markov import MarkovChain

State = Hashable


@dataclass(frozen=True)
class CTMCPath:
    """A simulated path represented by jump times and visited states.

    ``times[k]`` is the time at which ``states[k]`` is entered. The state is
    therefore constant between consecutive jump times, matching the
    holding-time construction used in the course.
    """

    times: np.ndarray
    states: tuple[State, ...]

    def state_at(self, t: float) -> State:
        """Return the state occupied at time ``t``."""
        if not np.isfinite(t) or t < 0:
            raise ValueError("t must be finite and non-negative")
        idx = int(np.searchsorted(self.times, t, side="right") - 1)
        idx = max(0, min(idx, len(self.states) - 1))
        return self.states[idx]


class ContinuousTimeMarkovChain:
    """Finite-state homogeneous CTMC represented by its generator ``Q``.

    For ``i != j`` the chapter defines ``p_ij(h)=q_ij h+o(h)``. Thus the
    off-diagonal entries of ``Q`` are non-negative, its diagonal entries are
    non-positive, and every row sums to zero. In the finite homogeneous case,
    ``P(t)=exp(tQ)``.
    """

    def __init__(
        self,
        generator: Sequence[Sequence[float]],
        states: Sequence[State] | None = None,
        *,
        tolerance: float = 1e-12,
    ) -> None:
        """Create a homogeneous CTMC from its infinitesimal generator ``Q``."""
        q = np.asarray(generator, dtype=float)
        if q.ndim != 2 or q.shape[0] != q.shape[1] or q.shape[0] == 0:
            raise ValueError("generator must be a non-empty square matrix")
        if not np.all(np.isfinite(q)):
            raise ValueError("generator must contain finite values")

        # These are the generator sign and row-sum conditions used in Chapter 3.
        off_diag = q - np.diag(np.diag(q))
        if np.any(off_diag < -tolerance):
            raise ValueError("off-diagonal generator entries must be non-negative")
        if np.any(np.diag(q) > tolerance):
            raise ValueError("generator diagonal entries must be non-positive")
        if not np.allclose(q.sum(axis=1), 0.0, atol=tolerance, rtol=0.0):
            raise ValueError("generator rows must sum to zero")

        labels = tuple(range(q.shape[0])) if states is None else tuple(states)
        if len(labels) != q.shape[0] or len(set(labels)) != len(labels):
            raise ValueError("states must be unique and match generator size")
        self._Q = q.copy()
        self._states = labels
        self._index = {s: i for i, s in enumerate(labels)}
        self._tolerance = tolerance

    @property
    def generator_matrix(self) -> np.ndarray:
        """Return a copy of the infinitesimal generator ``Q``."""
        return self._Q.copy()

    @property
    def states(self) -> tuple[State, ...]:
        """Return the ordered state labels."""
        return self._states

    @property
    def n_states(self) -> int:
        """Return the number of states."""
        return len(self._states)

    def infinitesimal_transition_matrix(self, h: float) -> np.ndarray:
        """Return the first-order matrix ``I+hQ``."""
        self._validate_time(h)
        return np.eye(self.n_states) + float(h) * self._Q

    def transition_matrix(self, t: float) -> np.ndarray:
        """Return ``P(t)=exp(tQ)`` for the finite homogeneous CTMC."""
        self._validate_time(t)
        return expm(self._Q * float(t))

    def transition_probability(self, source: State, target: State, t: float) -> float:
        """Return ``p_ij(t)=P(X_t=j | X_0=i)``."""
        return float(self.transition_matrix(t)[self._idx(source), self._idx(target)])

    def state_distribution(self, initial_distribution: Sequence[float], t: float) -> np.ndarray:
        """Return ``mu_t=mu_0 P(t)`` for a row-vector initial law."""
        mu = self._distribution(initial_distribution)
        return mu @ self.transition_matrix(t)

    def chapman_kolmogorov(self, s: float, t: float) -> np.ndarray:
        """Return ``P(s)P(t)=P(s+t)``."""
        self._validate_time(s)
        self._validate_time(t)
        return self.transition_matrix(s) @ self.transition_matrix(t)

    def forward_derivative(self, t: float) -> np.ndarray:
        """Return the forward Kolmogorov derivative ``P(t)Q``."""
        return self.transition_matrix(t) @ self._Q

    def backward_derivative(self, t: float) -> np.ndarray:
        """Return the backward Kolmogorov derivative ``QP(t)``."""
        return self._Q @ self.transition_matrix(t)

    def stationary_distribution(self) -> np.ndarray:
        """Return a stationary law solving ``pi Q=0`` and ``sum(pi)=1``."""
        n = self.n_states
        a = self._Q.T.copy()
        a[-1] = 1.0
        b = np.zeros(n)
        b[-1] = 1.0
        try:
            pi = np.linalg.solve(a, b)
        except np.linalg.LinAlgError as exc:
            raise ValueError("stationary distribution is not uniquely determined") from exc
        if np.any(pi < -1e-9):
            raise ValueError("no non-negative stationary distribution found")
        pi[np.abs(pi) < self._tolerance] = 0.0
        return pi / pi.sum()

    def long_run_cost(self, costs: Sequence[float]) -> float:
        """Return the stationary mean of a state-cost function."""
        costs = np.asarray(costs, dtype=float)
        if costs.shape != (self.n_states,) or not np.all(np.isfinite(costs)):
            raise ValueError("costs must have one finite value per state")
        return float(self.stationary_distribution() @ costs)

    def jump_chain_matrix(self) -> np.ndarray:
        """Return the transition matrix of the embedded jump chain.

        For ``q_i=-q_ii>0``, Chapter 3 gives ``p~_ij=q_ij/q_i`` for ``i!=j``.
        """
        transition = np.zeros_like(self._Q)
        for i in range(self.n_states):
            rate = -self._Q[i, i]
            if rate <= self._tolerance:
                transition[i, i] = 1.0
            else:
                transition[i] = self._Q[i] / rate
                transition[i, i] = 0.0
        return transition

    def jump_chain(self) -> MarkovChain:
        """Return the embedded discrete-time Markov chain."""
        return MarkovChain(self.jump_chain_matrix(), self._states, tolerance=self._tolerance)

    def holding_rate(self, state: State) -> float:
        """Return the exit rate ``q_i=-q_ii``."""
        return float(-self._Q[self._idx(state), self._idx(state)])

    def holding_time(self, state: State, *, rng: np.random.Generator | None = None) -> float:
        """Sample the exponential holding time in ``state``."""
        rate = self.holding_rate(state)
        if rate <= self._tolerance:
            return float("inf")
        generator = rng if rng is not None else np.random.default_rng()
        return float(generator.exponential(scale=1.0 / rate))

    def simulate(
        self,
        t_max: float,
        *,
        initial_state: State | None = None,
        rng: np.random.Generator | None = None,
    ) -> CTMCPath:
        """Simulate the holding-time/jump-chain construction up to ``t_max``."""
        self._validate_time(t_max)
        generator = rng if rng is not None else np.random.default_rng()
        current = 0 if initial_state is None else self._idx(initial_state)
        times = [0.0]
        states = [self._states[current]]
        elapsed = 0.0

        while elapsed < float(t_max):
            rate = -self._Q[current, current]
            if rate <= self._tolerance:
                break
            wait = float(generator.exponential(scale=1.0 / rate))
            if elapsed + wait > float(t_max):
                break
            elapsed += wait
            probabilities = self._Q[current].copy()
            probabilities[current] = 0.0
            probabilities /= rate
            current = int(generator.choice(self.n_states, p=probabilities))
            times.append(elapsed)
            states.append(self._states[current])
        return CTMCPath(np.asarray(times), tuple(states))

    def _idx(self, state: State) -> int:
        try:
            return self._index[state]
        except KeyError as exc:
            raise ValueError(f"unknown state: {state!r}") from exc

    def _distribution(self, values: Sequence[float]) -> np.ndarray:
        mu = np.asarray(values, dtype=float)
        if (
            mu.shape != (self.n_states,)
            or not np.all(np.isfinite(mu))
            or np.any(mu < 0)
            or not np.isclose(mu.sum(), 1.0)
        ):
            raise ValueError("initial distribution must be non-negative and sum to 1")
        return mu

    @staticmethod
    def _validate_time(t: float) -> None:
        if not np.isfinite(t) or t < 0:
            raise ValueError("time must be finite and non-negative")
