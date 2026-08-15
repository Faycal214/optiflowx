"""Homogeneous continuous-time Markov chains from MSPRO Chapter 3."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Hashable, Literal

import numpy as np
from scipy.linalg import expm

from .exceptions import GeneratorValidationError
from .markov_chain import MarkovChain
from .numerical.uniformization import uniformization_transition_matrix
from .validation import validate_generator, validate_probability_vector, validate_states, validate_tolerance

State = Hashable
TransitionMethod = Literal["expm", "uniformization"]


@dataclass(frozen=True)
class CTMCPath:
    """A simulated path represented by jump times and visited states."""

    times: np.ndarray
    states: tuple[State, ...]

    def state_at(self, t: float) -> State:
        """Return the state occupied at time ``t``."""
        if not np.isfinite(t) or t < 0:
            raise ValueError("t must be finite and non-negative")
        idx = int(np.searchsorted(self.times, t, side="right") - 1)
        idx = max(0, min(idx, len(self.states) - 1))
        return self.states[idx]

    def occupation_time(self, state: State, horizon: float) -> float:
        """Return the time spent in ``state`` on ``[0, horizon]``."""
        if not np.isfinite(horizon) or horizon < 0:
            raise ValueError("horizon must be finite and non-negative")
        if len(self.times) == 0 or self.times[0] != 0.0:
            raise ValueError("CTMC path must start at time 0")
        if horizon == 0:
            return 0.0
        total = 0.0
        times = np.asarray(self.times, dtype=float)
        for i, start in enumerate(times):
            if start >= horizon:
                break
            end = horizon if i + 1 == len(times) else min(float(times[i + 1]), horizon)
            if self.states[i] == state:
                total += max(0.0, end - float(start))
        return float(total)

    def occupation_fraction(self, state: State, horizon: float) -> float:
        """Return the occupation fraction of ``state`` on ``[0, horizon]``."""
        if horizon <= 0 or not np.isfinite(horizon):
            raise ValueError("horizon must be strictly positive and finite")
        return self.occupation_time(state, horizon) / float(horizon)


class ContinuousTimeMarkovChain:
    """Finite-state homogeneous CTMC represented by its generator ``Q``."""

    def __init__(
        self,
        generator: Sequence[Sequence[float]],
        states: Sequence[State] | None = None,
        *,
        tolerance: float = 1e-12,
    ) -> None:
        """Create a homogeneous CTMC from its infinitesimal generator ``Q``."""
        self._tolerance = validate_tolerance(tolerance)
        q = validate_generator(generator, tolerance=self._tolerance)
        labels = validate_states(states, q.shape[0])
        self._Q = q
        self._states = labels
        self._index = {s: i for i, s in enumerate(labels)}

    @property
    def generator_matrix(self) -> np.ndarray:
        """Return a copy of the infinitesimal generator ``Q``."""
        return self._Q.copy()

    @property
    def generator(self) -> np.ndarray:
        """Return the infinitesimal generator ``Q``."""
        return self.generator_matrix

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

    def transition_matrix(
        self,
        t: float,
        *,
        method: TransitionMethod = "expm",
        max_terms: int = 100_000,
    ) -> np.ndarray:
        """Return ``P(t)`` using SciPy ``expm`` or CTMC uniformization.

        Parameters
        ----------
        t:
            Non-negative time.
        method:
            ``"expm"`` uses the matrix exponential. ``"uniformization"``
            uses Jensen's uniformization series.
        max_terms:
            Maximum Poisson-series terms for uniformization.
        """
        self._validate_time(t)
        if method == "expm":
            return expm(self._Q * float(t))
        if method == "uniformization":
            return uniformization_transition_matrix(
                self._Q,
                t,
                tolerance=self._tolerance,
                max_terms=max_terms,
            )
        raise ValueError("method must be 'expm' or 'uniformization'")

    def transition_matrix_at(
        self,
        t: float,
        *,
        method: TransitionMethod = "expm",
        max_terms: int = 100_000,
    ) -> np.ndarray:
        """Return ``P(t)`` using the canonical transition-matrix API."""
        return self.transition_matrix(t, method=method, max_terms=max_terms)

    def transition_matrix_uniformized(self, t: float, *, max_terms: int = 100_000) -> np.ndarray:
        """Return ``P(t)`` explicitly through Jensen uniformization."""
        return self.transition_matrix(t, method="uniformization", max_terms=max_terms)

    def transition_probability(self, source: State, target: State, t: float) -> float:
        """Return ``p_ij(t)=P(X_t=j | X_0=i)``."""
        return float(self.transition_matrix(t)[self._idx(source), self._idx(target)])

    def state_distribution(self, initial_distribution: Sequence[float], t: float) -> np.ndarray:
        """Return ``mu_t=mu_0 P(t)`` for a row-vector initial law."""
        mu = validate_probability_vector(initial_distribution, self.n_states, tolerance=self._tolerance, name="initial_distribution")
        return mu @ self.transition_matrix(t)

    def chapman_kolmogorov(self, s: float, t: float) -> np.ndarray:
        """Return ``P(s)P(t)=P(s+t)``."""
        self._validate_time(s)
        self._validate_time(t)
        return self.transition_matrix(s) @ self.transition_matrix(t)

    def forward_derivative(self, t: float) -> np.ndarray:
        """Return the forward Kolmogorov derivative ``P(t)Q``."""
        return self.transition_matrix(t) @ self._Q

    def forward_equation(self, t: float) -> np.ndarray:
        """Return the forward Kolmogorov derivative ``P(t)Q``."""
        return self.forward_derivative(t)

    def backward_derivative(self, t: float) -> np.ndarray:
        """Return the backward Kolmogorov derivative ``QP(t)``."""
        return self._Q @ self.transition_matrix(t)

    def backward_equation(self, t: float) -> np.ndarray:
        """Return the backward Kolmogorov derivative ``QP(t)``."""
        return self.backward_derivative(t)

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

    def communicating_classes(self) -> list[tuple[State, ...]]:
        """Return communication classes through the embedded jump chain."""
        return self.jump_chain().communicating_classes()

    def stationary_distribution_from_jump_chain(self) -> np.ndarray:
        """Recover the CTMC stationary law from its jump-chain stationary law."""
        rates = -np.diag(self._Q)
        if np.any(rates <= self._tolerance):
            raise GeneratorValidationError("positive holding rates are required in every state")
        phi = self.jump_chain().stationary_distribution()
        weights = phi / rates
        return weights / weights.sum()

    def mean_return_time(self, state: State) -> float:
        """Return the mean continuous-time return time for ``state``."""
        pi = self.stationary_distribution()
        i = self._idx(state)
        rate = -self._Q[i, i]
        if rate <= self._tolerance or pi[i] <= 0:
            return float("inf")
        return float(1.0 / (rate * pi[i]))

    def long_run_cost(self, costs: Sequence[float]) -> float:
        """Return the stationary mean of a state-cost function."""
        costs = np.asarray(costs, dtype=float)
        if costs.shape != (self.n_states,) or not np.all(np.isfinite(costs)):
            raise ValueError("costs must have one finite value per state")
        return float(self.stationary_distribution() @ costs)

    def jump_chain_matrix(self) -> np.ndarray:
        """Return the transition matrix of the embedded jump chain."""
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

    @staticmethod
    def _validate_time(t: float) -> None:
        if not np.isfinite(t) or t < 0:
            raise ValueError("time must be finite and non-negative")


__all__ = ["CTMCPath", "ContinuousTimeMarkovChain"]
