"""Discrete-time Markov chains (CMTD)."""

from __future__ import annotations

from collections.abc import Sequence
from math import gcd
from typing import Hashable

import numpy as np

State = Hashable


class MarkovChain:
    """Finite-state, time-homogeneous discrete-time Markov chain.

    The API follows the CMTD material in the MSPRO course: transition
    matrices, state distributions, Chapman-Kolmogorov, first visits,
    communication classes, recurrence/transience, periodicity, ergodicity,
    stationary and limiting distributions, absorption probabilities, and
    simulation.
    """

    def __init__(
        self,
        transition_matrix: Sequence[Sequence[float]],
        states: Sequence[State] | None = None,
        *,
        tolerance: float = 1e-12,
    ) -> None:
        matrix = np.asarray(transition_matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("transition_matrix must be a square 2-D matrix")
        if matrix.shape[0] == 0:
            raise ValueError("transition_matrix must contain at least one state")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("transition_matrix must contain finite values")
        if np.any(matrix < -tolerance):
            raise ValueError("transition probabilities must be non-negative")
        if not np.allclose(matrix.sum(axis=1), 1.0, atol=tolerance, rtol=0.0):
            raise ValueError("each row of transition_matrix must sum to 1")
        matrix = matrix.copy()
        matrix[np.abs(matrix) < tolerance] = 0.0
        matrix[matrix < 0.0] = 0.0

        if states is None:
            state_labels: tuple[State, ...] = tuple(range(matrix.shape[0]))
        else:
            state_labels = tuple(states)
            if len(state_labels) != matrix.shape[0]:
                raise ValueError("states must have the same length as the matrix")
            if len(set(state_labels)) != len(state_labels):
                raise ValueError("states must be unique")

        self._P = matrix
        self._states = state_labels
        self._index = {state: i for i, state in enumerate(state_labels)}
        self._tolerance = tolerance

    @property
    def transition_matrix(self) -> np.ndarray:
        """Return a copy of the one-step transition matrix ``P``."""
        return self._P.copy()

    @property
    def states(self) -> tuple[State, ...]:
        """Return the ordered state labels."""
        return self._states

    @property
    def n_states(self) -> int:
        """Return the number of states."""
        return len(self._states)

    def n_step_transition(self, n: int) -> np.ndarray:
        """Return the n-step transition matrix ``P^(n) = P^n``."""
        self._validate_nonnegative_integer(n, "n")
        return np.linalg.matrix_power(self._P, int(n))

    def state_distribution(
        self,
        initial_distribution: Sequence[float],
        n: int,
    ) -> np.ndarray:
        """Return ``mu_n = mu_0 P^n`` for an initial law ``mu_0``."""
        self._validate_nonnegative_integer(n, "n")
        mu = self._validate_distribution(initial_distribution, "initial_distribution")
        return mu @ self.n_step_transition(int(n))

    def chapman_kolmogorov(self, m: int, n: int) -> np.ndarray:
        """Return ``P^(m+n)`` through the Chapman-Kolmogorov product."""
        self._validate_nonnegative_integer(m, "m")
        self._validate_nonnegative_integer(n, "n")
        return self.n_step_transition(int(m)) @ self.n_step_transition(int(n))

    def transition_graph(self) -> dict[State, tuple[State, ...]]:
        """Return the directed graph induced by positive transitions."""
        return {
            state: tuple(
                self._states[j]
                for j, probability in enumerate(self._P[i])
                if probability > self._tolerance
            )
            for i, state in enumerate(self._states)
        }

    def accessible(self, source: State, target: State) -> bool:
        """Return whether ``target`` is accessible from ``source``."""
        source_idx = self._state_index(source)
        target_idx = self._state_index(target)
        reachable = {source_idx}
        frontier = [source_idx]
        while frontier:
            current = frontier.pop()
            for nxt, probability in enumerate(self._P[current]):
                if probability > self._tolerance and nxt not in reachable:
                    reachable.add(nxt)
                    frontier.append(nxt)
        return target_idx in reachable

    def communicate(self, source: State, target: State) -> bool:
        """Return whether two states communicate."""
        return self.accessible(source, target) and self.accessible(target, source)

    def communicating_classes(self) -> list[tuple[State, ...]]:
        """Return the communicating classes of the finite chain."""
        graph = [
            [j for j, probability in enumerate(row) if probability > self._tolerance]
            for row in self._P
        ]
        reverse = [[] for _ in range(self.n_states)]
        for i, neighbors in enumerate(graph):
            for j in neighbors:
                reverse[j].append(i)

        visited = [False] * self.n_states
        order: list[int] = []

        def dfs(vertex: int) -> None:
            visited[vertex] = True
            for neighbor in graph[vertex]:
                if not visited[neighbor]:
                    dfs(neighbor)
            order.append(vertex)

        for vertex in range(self.n_states):
            if not visited[vertex]:
                dfs(vertex)

        visited = [False] * self.n_states
        components: list[tuple[State, ...]] = []

        def reverse_dfs(vertex: int, component: list[int]) -> None:
            visited[vertex] = True
            component.append(vertex)
            for neighbor in reverse[vertex]:
                if not visited[neighbor]:
                    reverse_dfs(neighbor, component)

        for vertex in reversed(order):
            if not visited[vertex]:
                component: list[int] = []
                reverse_dfs(vertex, component)
                components.append(tuple(self._states[i] for i in component))

        return components

    def is_irreducible(self) -> bool:
        """Return whether the state space is one communication class."""
        return len(self.communicating_classes()) == 1

    def closed_classes(self) -> list[tuple[State, ...]]:
        """Return closed (absorbing) communicating classes."""
        graph = self.transition_graph()
        closed: list[tuple[State, ...]] = []
        for component in self.communicating_classes():
            members = set(component)
            if all(set(graph[state]).issubset(members) for state in component):
                closed.append(component)
        return closed

    def is_absorbing_state(self, state: State) -> bool:
        """Return whether ``p_ii = 1`` for the given state."""
        i = self._state_index(state)
        return bool(abs(self._P[i, i] - 1.0) <= self._tolerance)

    def classify_states(self) -> dict[State, str]:
        """Classify states as recurrent or transient."""
        graph = [
            {j for j, probability in enumerate(row) if probability > self._tolerance}
            for row in self._P
        ]
        classifications: dict[State, str] = {}
        for component in self.communicating_classes():
            indices = {self._index[state] for state in component}
            closed = all(graph[i].issubset(indices) for i in indices)
            label = "recurrent" if closed else "transient"
            for state in component:
                classifications[state] = label
        return classifications

    def period(self, state: State) -> int | float:
        """Return the period ``d(i)`` of a state."""
        root = self._state_index(state)
        graph = [
            [j for j, probability in enumerate(row) if probability > self._tolerance]
            for row in self._P
        ]
        distances = {root: 0}
        queue = [root]
        period = 0
        while queue:
            u = queue.pop(0)
            for v in graph[u]:
                if v not in distances:
                    distances[v] = distances[u] + 1
                    queue.append(v)
                else:
                    period = gcd(period, distances[u] + 1 - distances[v])
        return float("inf") if period == 0 else abs(period)

    def is_aperiodic(self) -> bool:
        """Return whether every state is aperiodic."""
        return all(self.period(state) == 1 for state in self._states)

    def is_ergodic(self) -> bool:
        """Return whether every state is recurrent and aperiodic."""
        classifications = self.classify_states()
        return all(
            classifications[state] == "recurrent" and self.period(state) == 1
            for state in self._states
        )

    def first_visit_probability(self, source: State, target: State, n: int) -> float:
        """Return ``f_ij^(n)``, the probability of first visiting j at n."""
        self._validate_positive_integer(n, "n")
        i = self._state_index(source)
        j = self._state_index(target)
        killed = self._P.copy()
        killed[j, :] = 0.0
        distribution = np.zeros(self.n_states, dtype=float)
        distribution[i] = 1.0
        for step in range(1, int(n)):
            distribution = distribution @ killed
        return float((distribution @ self._P)[j])

    def visit_probability(self, source: State, target: State, n: int) -> float:
        """Return ``sum_{k=1}^n f_ij^(k)``."""
        self._validate_positive_integer(n, "n")
        return float(
            sum(self.first_visit_probability(source, target, k) for k in range(1, int(n) + 1))
        )

    def expected_hitting_time(self, source: State, target: State) -> float:
        """Return ``E(T_ij | X_0=i)`` when the hit is almost sure, else infinity."""
        i = self._state_index(source)
        j = self._state_index(target)
        if i == j:
            return 0.0
        if not self._hit_is_almost_surely(source, target):
            return float("inf")
        transient = [k for k in range(self.n_states) if k != j]
        A = np.eye(len(transient)) - self._P[np.ix_(transient, transient)]
        b = np.ones(len(transient))
        values = np.linalg.solve(A, b)
        return float(values[transient.index(i)])

    def stationary_distribution(self) -> np.ndarray:
        """Return the unique stationary law for an irreducible finite chain."""
        if not self.is_irreducible():
            raise ValueError(
                "stationary_distribution() requires an irreducible finite chain"
            )
        return self._stationary_for_matrix(self._P)

    def limiting_distribution(self) -> np.ndarray:
        """Return the limiting distribution matrix under the course conditions."""
        if self.is_ergodic():
            pi = self.stationary_distribution()
            return np.tile(pi, (self.n_states, 1))

        closed = self.closed_classes()
        classifications = self.classify_states()
        if len(closed) != 1 or any(
            classifications[state] != "transient" for state in self._states if state not in closed[0]
        ):
            raise ValueError("the course conditions for a limiting distribution are not met")

        indices = [self._index[state] for state in closed[0]]
        subchain = self._P[np.ix_(indices, indices)]
        if not self._subchain_is_ergodic(subchain):
            raise ValueError("the unique closed class is not ergodic")
        pi_class = self._stationary_for_matrix(subchain)
        result = np.zeros((self.n_states, self.n_states))
        result[:, indices] = pi_class
        return result

    def absorption_probability(
        self,
        source: State,
        absorbing_class: Sequence[State],
    ) -> float:
        """Return the probability of eventual absorption in a closed class."""
        target = set(absorbing_class)
        if not target:
            raise ValueError("absorbing_class must not be empty")
        if not all(state in self._index for state in target):
            raise ValueError("absorbing_class contains an unknown state")
        if not any(set(cls) == target for cls in self.closed_classes()):
            raise ValueError("absorbing_class must be a closed communicating class")

        source_idx = self._state_index(source)
        target_indices = {self._index[state] for state in target}
        if source_idx in target_indices:
            return 1.0

        transient = [i for i in range(self.n_states) if i not in target_indices]
        A = np.eye(len(transient)) - self._P[np.ix_(transient, transient)]
        b = self._P[np.ix_(transient, sorted(target_indices))].sum(axis=1)
        values = np.linalg.solve(A, b)
        return float(values[transient.index(source_idx)])

    def simulate(
        self,
        n_steps: int,
        *,
        initial_state: State | None = None,
        initial_distribution: Sequence[float] | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[State]:
        """Simulate one trajectory ``X_0, ..., X_n``."""
        self._validate_nonnegative_integer(n_steps, "n_steps")
        if initial_state is not None and initial_distribution is not None:
            raise ValueError("provide only one initial condition")

        generator = rng if rng is not None else np.random.default_rng()
        if initial_distribution is not None:
            distribution = self._validate_distribution(initial_distribution, "initial_distribution")
            current = int(generator.choice(self.n_states, p=distribution))
        elif initial_state is None:
            current = 0
        else:
            current = self._state_index(initial_state)

        trajectory = [self._states[current]]
        for _ in range(int(n_steps)):
            current = int(generator.choice(self.n_states, p=self._P[current]))
            trajectory.append(self._states[current])
        return trajectory

    def _state_index(self, state: State) -> int:
        try:
            return self._index[state]
        except KeyError as exc:
            raise ValueError(f"unknown state: {state!r}") from exc

    def _hit_is_almost_surely(self, source: State, target: State) -> bool:
        source_idx = self._state_index(source)
        target_idx = self._state_index(target)
        if source_idx == target_idx:
            return True
        remaining = {i for i in range(self.n_states) if i != target_idx}
        changed = True
        while changed:
            changed = False
            for state in tuple(remaining):
                if any(
                    self._P[state, nxt] > self._tolerance and nxt in remaining
                    for nxt in range(self.n_states)
                ):
                    continue
                remaining.remove(state)
                changed = True
        return source_idx not in remaining

    @staticmethod
    def _stationary_for_matrix(matrix: np.ndarray) -> np.ndarray:
        n = matrix.shape[0]
        A = matrix.T - np.eye(n)
        A[-1] = 1.0
        b = np.zeros(n)
        b[-1] = 1.0
        pi = np.linalg.solve(A, b)
        pi[np.abs(pi) < 1e-12] = 0.0
        return pi / pi.sum()

    @staticmethod
    def _subchain_is_ergodic(matrix: np.ndarray) -> bool:
        n = matrix.shape[0]
        if n == 1:
            return True
        reach = (matrix > 0).astype(int)
        power = reach.copy()
        for _ in range(n - 1):
            power = power @ reach
        if np.any(power == 0):
            return False
        for i in range(n):
            period = 0
            power_matrix = matrix.copy()
            for k in range(1, n * n + 1):
                if power_matrix[i, i] > 1e-12:
                    period = gcd(period, k)
                power_matrix = power_matrix @ matrix
            if period != 1:
                return False
        return True

    @staticmethod
    def _validate_nonnegative_integer(value: int, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be a non-negative integer")
        if value < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    @classmethod
    def _validate_positive_integer(cls, value: int, name: str) -> None:
        cls._validate_nonnegative_integer(value, name)
        if value == 0:
            raise ValueError(f"{name} must be positive")

    def _validate_distribution(self, distribution: Sequence[float], name: str) -> np.ndarray:
        values = np.asarray(distribution, dtype=float)
        if values.shape != (self.n_states,):
            raise ValueError(f"{name} has the wrong shape")
        if np.any(~np.isfinite(values)) or np.any(values < -self._tolerance):
            raise ValueError(f"{name} must contain non-negative finite values")
        if not np.isclose(values.sum(), 1.0, atol=self._tolerance, rtol=0.0):
            raise ValueError(f"{name} must sum to 1")
        values = values.copy()
        values[np.abs(values) < self._tolerance] = 0.0
        return values / values.sum()
