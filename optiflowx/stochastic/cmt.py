"""Course-level operations for discrete-time Markov chains (CMTD)."""

from __future__ import annotations

from math import gcd
from collections.abc import Sequence
from typing import Hashable

import numpy as np

from .markov import MarkovChain

State = Hashable


class CMTD(MarkovChain):
    """Finite-state CMTD operations matching Chapter 1 of the course."""

    def set_initial_distribution(self, distribution: Sequence[float]) -> np.ndarray:
        mu = np.asarray(distribution, dtype=float)
        if mu.shape != (self.n_states,):
            raise ValueError("initial distribution has the wrong shape")
        if not np.all(np.isfinite(mu)) or np.any(mu < 0) or not np.isclose(mu.sum(), 1.0):
            raise ValueError("initial distribution must be non-negative and sum to 1")
        self._mu0 = mu.copy()
        return self._mu0.copy()

    @property
    def initial_distribution(self) -> np.ndarray | None:
        return None if not hasattr(self, "_mu0") else self._mu0.copy()

    def state_distribution(self, n: int, initial_distribution: Sequence[float] | None = None) -> np.ndarray:
        """Return mu_n = mu_0 P^n."""
        mu = np.asarray(initial_distribution if initial_distribution is not None else self._mu0, dtype=float) \
            if (initial_distribution is not None or hasattr(self, "_mu0")) else None
        if mu is None:
            raise ValueError("an initial distribution is required")
        self.set_initial_distribution(mu)
        if n < 0 or isinstance(n, bool) or not isinstance(n, (int, np.integer)):
            raise ValueError("n must be a non-negative integer")
        return mu @ self.n_step_transition(int(n))

    def chapman_kolmogorov(self, m: int, n: int) -> np.ndarray:
        """Return P^(m+n), equal to P^m P^n."""
        if m < 0 or n < 0 or any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in (m, n)):
            raise ValueError("m and n must be non-negative integers")
        return self.n_step_transition(int(m)) @ self.n_step_transition(int(n))

    def first_visit_probabilities(self, target: State, n_steps: int) -> np.ndarray:
        """Return f_ij^(n), the probability of first visiting target at step n, for every i."""
        if n_steps < 1 or isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)):
            raise ValueError("n_steps must be a positive integer")
        j = self._index[target]
        f = np.zeros(self.n_states)
        for i in range(self.n_states):
            if i == j and n_steps == 1:
                f[i] = self._P[i, j]
            else:
                value = self.n_step_transition(int(n_steps))[i, j]
                for k in range(1, int(n_steps)):
                    value -= self.first_visit_probabilities(target, k)[i] * self.n_step_transition(int(n_steps - k))[j, j]
                f[i] = max(0.0, value)
        return f

    def visit_probability(self, start: State, target: State, max_steps: int = 10000) -> float:
        """Numerically accumulate first-visit probabilities through max_steps."""
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        i = self._index[start]
        j = self._index[target]
        if i == j:
            return 1.0
        # Dynamic first-hit recursion, avoiding repeated recursive recomputation.
        p = self._P
        powers = [np.eye(self.n_states)]
        for _ in range(max_steps):
            powers.append(powers[-1] @ p)
        first = np.zeros(max_steps + 1)
        for n in range(1, max_steps + 1):
            value = powers[n][i, j]
            for k in range(1, n):
                value -= first[k] * powers[n - k][j, j]
            first[n] = max(0.0, value)
        return float(min(1.0, first.sum()))

    def period(self, state: State) -> int:
        """Return d(i)=gcd{n>=1: p_ii^(n)>0}; infinity is represented by 0."""
        root = self._index[state]
        graph = [[j for j, p in enumerate(row) if p > self._tolerance] for row in self._P]
        distance = {root: 0}
        stack = [root]
        period = 0
        while stack:
            u = stack.pop()
            for v in graph[u]:
                if v not in distance:
                    distance[v] = distance[u] + 1
                    stack.append(v)
                if v == root:
                    period = gcd(period, distance[u] + 1)
        return period

    def is_ergodic(self) -> bool:
        """Return whether every state is positive recurrent and aperiodic."""
        if not all(label == "recurrent" for label in self.classify_states().values()):
            return False
        return all(self.period(state) == 1 for state in self.states)

    def mean_hitting_time(self, start: State, target: State) -> float:
        """Return E[T_ij] for i != j when the target is hit almost surely."""
        i = self._index[start]
        j = self._index[target]
        if i == j:
            raise ValueError("start and target must differ for a hitting time")
        if self.hitting_probability(start, target) < 1.0 - 1e-10:
            return float("inf")
        transient = [k for k in range(self.n_states) if k != j]
        q = self._P[np.ix_(transient, transient)]
        h = np.linalg.solve(np.eye(len(transient)) - q, np.ones(len(transient)))
        return float(h[transient.index(i)])

    def mean_return_time(self, state: State) -> float:
        """Return E[T_ii], the mean first return time when finite."""
        idx = self._index[state]
        if self.classify_states()[state] == "transient":
            return float("inf")
        if self.is_irreducible():
            pi = self.stationary_distribution()
            return float(1.0 / pi[idx])
        # First-return system: h_i = 1 + sum_j p_ij h_j with the target's
        # equation excluding the immediate return contribution.
        a = np.eye(self.n_states)
        b = np.ones(self.n_states)
        for row in range(self.n_states):
            if row == idx:
                a[row] -= self._P[row]
                a[row, idx] += self._P[row, idx]
            else:
                a[row] -= self._P[row]
                a[row, idx] = 0.0
        try:
            return float(np.linalg.solve(a, b)[idx])
        except np.linalg.LinAlgError:
            return float("inf")

    def hitting_probability(self, start: State, target: State) -> float:
        """Return P_i(T_ij < infinity) for finite-state CMTD."""
        i = self._index[start]
        j = self._index[target]
        if i == j:
            return 1.0
        reachable = self._reachable_from(i)
        if j not in reachable:
            return 0.0
        states = sorted(reachable - {j})
        if not states:
            return 1.0
        q = self._P[np.ix_(states, states)]
        r = self._P[np.ix_(states, [j])].ravel()
        values = np.linalg.solve(np.eye(len(states)) - q, r)
        return float(values[states.index(i)])

    def limiting_transition_matrix(self, tol: float = 1e-12, max_steps: int = 100000) -> np.ndarray:
        """Numerically return lim P^n when the Chapter 1 limit exists."""
        if not self.is_ergodic():
            closed = [c for c in self.communicating_classes() if all(self.classify_states()[s] == "recurrent" for s in c)]
            if len(closed) != 1 or any(self.period(s) != 1 for s in closed[0]):
                raise ValueError("the chapter's limiting-distribution conditions are not satisfied")
        power = np.eye(self.n_states)
        previous = None
        for _ in range(max_steps):
            power = power @ self._P
            if previous is not None and np.max(np.abs(power - previous)) < tol:
                return power
            previous = power.copy()
        raise RuntimeError("limit did not converge within max_steps")

    def limiting_distribution(self, initial_distribution: Sequence[float] | None = None, tol: float = 1e-12) -> np.ndarray:
        """Return the limiting state distribution under the course conditions."""
        mu = np.asarray(initial_distribution if initial_distribution is not None else self._mu0, dtype=float) \
            if (initial_distribution is not None or hasattr(self, "_mu0")) else None
        if mu is None:
            raise ValueError("an initial distribution is required")
        return mu @ self.limiting_transition_matrix(tol=tol)

    def absorption_probability(self, start: State, target_class: Sequence[State]) -> float:
        """Return the absorption probability into a specified closed ergodic class."""
        targets = tuple(target_class)
        if not targets:
            raise ValueError("target_class cannot be empty")
        target_idx = {self._index[s] for s in targets}
        i = self._index[start]
        if i in target_idx:
            return 1.0
        recurrent = self.classify_states()
        if any(recurrent[s] != "recurrent" for s in targets):
            raise ValueError("target_class must be a closed recurrent class")
        trans = [k for k, s in enumerate(self.states) if recurrent[s] == "transient"]
        if i not in trans:
            return 0.0
        a = np.eye(len(trans))
        b = np.zeros(len(trans))
        for r, u in enumerate(trans):
            a[r] -= self._P[u, trans]
            b[r] = self._P[u, list(target_idx)].sum()
        values = np.linalg.solve(a, b)
        return float(values[trans.index(i)])

    def _reachable_from(self, start: int) -> set[int]:
        graph = [[j for j, p in enumerate(row) if p > self._tolerance] for row in self._P]
        seen = {start}
        stack = [start]
        while stack:
            u = stack.pop()
            for v in graph[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return seen
