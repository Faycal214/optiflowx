"""Discrete-time Markov chains."""

from __future__ import annotations

from collections.abc import Sequence
from math import gcd
from typing import Hashable

import numpy as np

State = Hashable


class MarkovChain:
    """A finite-state, time-homogeneous discrete-time Markov chain.

    A chain is characterized by a stochastic transition matrix ``P`` and,
    when simulation is requested, an initial state or initial distribution.
    The implementation follows the finite-state CMTD setting used in the
    MSPRO Processus Aléatoires course.
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
        """Return the ``n``-step transition matrix ``P^n``."""

        if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
            raise TypeError("n must be a non-negative integer")
        if n < 0:
            raise ValueError("n must be non-negative")
        return np.linalg.matrix_power(self._P, int(n))

    def stationary_distribution(self) -> np.ndarray:
        """Return a stationary distribution for an irreducible finite chain.

        For reducible chains there can be multiple stationary distributions;
        in that case this method raises ``ValueError`` rather than silently
        selecting one.
        """

        if not self.is_irreducible():
            raise ValueError(
                "a reducible finite chain may have multiple stationary "
                "distributions; select a closed communicating class"
            )

        n = self.n_states
        a = self._P.T - np.eye(n)
        a[-1] = 1.0
        b = np.zeros(n)
        b[-1] = 1.0
        pi = np.linalg.solve(a, b)
        pi[np.abs(pi) < self._tolerance] = 0.0
        return pi / pi.sum()

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

    def communicating_classes(self) -> list[tuple[State, ...]]:
        """Return the communicating classes of the finite chain."""

        graph = [
            [j for j, p in enumerate(row) if p > self._tolerance]
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
        """Return whether every state communicates with every other state."""

        return len(self.communicating_classes()) == 1

    def is_aperiodic(self) -> bool:
        """Return whether the finite chain is aperiodic.

        For a reducible chain, this returns ``True`` only when every closed
        communicating class is aperiodic and singleton transient classes do
        not introduce periodic behavior. For the usual irreducible CMTD case,
        it is the standard period test.
        """

        graph = [
            [j for j, p in enumerate(row) if p > self._tolerance]
            for row in self._P
        ]
        classes = self.communicating_classes()

        for state_class in classes:
            indices = [self._index[s] for s in state_class]
            if len(indices) == 1 and indices[0] not in graph[indices[0]]:
                continue

            root = indices[0]
            distance = {root: 0}
            stack = [root]
            period = 0
            allowed = set(indices)
            while stack:
                u = stack.pop()
                for v in graph[u]:
                    if v not in allowed:
                        continue
                    if v not in distance:
                        distance[v] = distance[u] + 1
                        stack.append(v)
                    else:
                        period = gcd(period, distance[u] + 1 - distance[v])

            if abs(period) != 1:
                return False

        return True

    def classify_states(self) -> dict[State, str]:
        """Classify finite-chain states as ``recurrent`` or ``transient``."""

        graph = [
            {j for j, p in enumerate(row) if p > self._tolerance}
            for row in self._P
        ]
        classifications: dict[State, str] = {}
        for component in self.communicating_classes():
            indices = {self._index[s] for s in component}
            closed = all(graph[i].issubset(indices) for i in indices)
            label = "recurrent" if closed else "transient"
            for state in component:
                classifications[state] = label
        return classifications

    def simulate(
        self,
        n_steps: int,
        *,
        initial_state: State | None = None,
        initial_distribution: Sequence[float] | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[State]:
        """Simulate one trajectory ``X_0, ..., X_n``.

        Exactly one of ``initial_state`` and ``initial_distribution`` may be
        supplied. If neither is given, the default is state 0.
        """

        if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)):
            raise TypeError("n_steps must be a non-negative integer")
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        if initial_state is not None and initial_distribution is not None:
            raise ValueError("provide only one initial condition")

        generator = rng if rng is not None else np.random.default_rng()

        if initial_distribution is not None:
            distribution = np.asarray(initial_distribution, dtype=float)
            if distribution.shape != (self.n_states,):
                raise ValueError("initial_distribution has the wrong shape")
            if np.any(distribution < 0) or not np.isclose(distribution.sum(), 1.0):
                raise ValueError("initial_distribution must be non-negative and sum to 1")
            current = int(generator.choice(self.n_states, p=distribution))
        elif initial_state is None:
            current = 0
        else:
            try:
                current = self._index[initial_state]
            except KeyError as exc:
                raise ValueError(f"unknown initial state: {initial_state!r}") from exc

        trajectory = [self._states[current]]
        for _ in range(int(n_steps)):
            current = int(generator.choice(self.n_states, p=self._P[current]))
            trajectory.append(self._states[current])
        return trajectory
