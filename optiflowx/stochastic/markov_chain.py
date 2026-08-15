"""Discrete-time Markov chains (CMTD) from MSPRO Chapter 1."""

from __future__ import annotations

from collections.abc import Sequence
from math import gcd
from typing import Hashable

import numpy as np

from .validation import validate_states, validate_stochastic_matrix, validate_tolerance

State = Hashable


class MarkovChain:
    """Finite-state homogeneous discrete-time Markov chain.
    
    Mathematical object
    ------------------
    Public stochastic object exposed by the OptiFlowX API.
    
    Course basis
    ------------
    The implementation follows the corresponding MSPRO course material documented by OptiFlowX.
    
    Parameters
    ----------
    transition_matrix : Sequence[Sequence[float]]
        Transition probability matrix.
    states : Sequence[State] | None
        State labels in matrix order.
    tolerance : float, default None
        Numerical tolerance.
    
    Examples
    --------
    See the executable examples for `markov_chain.py` and the API reference."""

    def __init__(self, transition_matrix: Sequence[Sequence[float]], states: Sequence[State] | None = None, *, tolerance: float = 1e-12) -> None:
        """Public stochastic operation.
        
        Parameters
        ----------
        transition_matrix : Sequence[Sequence[float]]
            Transition probability matrix.
        states : Sequence[State] | None
            State labels in matrix order.
        tolerance : float, default None
            Numerical tolerance.
        
        Examples
        --------
        See the executable examples and API reference for `__init__`."""
        tolerance = validate_tolerance(tolerance)
        matrix = validate_stochastic_matrix(transition_matrix, tolerance=tolerance)
        labels = validate_states(states, matrix.shape[0])
        self._P = matrix
        self._states = labels
        self._index = {state: i for i, state in enumerate(labels)}
        self._tolerance = tolerance

    @property
    def transition_matrix(self) -> np.ndarray:
        """Return the one-step transition matrix ``P`` as a copy.
        
        
        Returns
        -------
        np.ndarray
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `transition_matrix`."""
        return self._P.copy()

    @property
    def states(self) -> tuple[State, ...]:
        """Return the ordered state labels.
        
        
        Returns
        -------
        tuple[State, ...]
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `states`."""
        return self._states

    @property
    def n_states(self) -> int:
        """Return the number of states.
        
        
        Returns
        -------
        int
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `n_states`."""
        return len(self._states)

    def n_step_transition(self, n: int) -> np.ndarray:
        """Compute the ``n``-step transition matrix ``P^n``.
        
        Parameters
        ----------
        n : int
            Integer index required by the operation.
        
        Returns
        -------
        np.ndarray
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `n_step_transition`."""
        self._nonnegative_int(n, "n")
        return np.linalg.matrix_power(self._P, int(n))

    def transition_matrix_at(self, n: int) -> np.ndarray:
        """Return the canonical ``n``-step transition matrix ``P^n``.
        
        Parameters
        ----------
        n : int
            Integer index required by the operation.
        
        Returns
        -------
        np.ndarray
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `transition_matrix_at`."""
        return self.n_step_transition(n)

    def state_distribution(self, initial_distribution: Sequence[float], n: int) -> np.ndarray:
        """Return ``mu_n = mu_0 P^n``.
        
        Parameters
        ----------
        initial_distribution : Sequence[float]
            Initial probability distribution.
        n : int
            Integer index required by the operation.
        
        Returns
        -------
        np.ndarray
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `state_distribution`."""
        self._nonnegative_int(n, "n")
        return self._distribution(initial_distribution) @ self.n_step_transition(n)

    def chapman_kolmogorov(self, m: int, n: int) -> np.ndarray:
        """Evaluate ``P^m P^n = P^(m+n)``.
        
        Parameters
        ----------
        m : int
            Non-negative integer time step.
        n : int
            Integer index required by the operation.
        
        Returns
        -------
        np.ndarray
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `chapman_kolmogorov`."""
        self._nonnegative_int(m, "m")
        self._nonnegative_int(n, "n")
        return self.n_step_transition(m) @ self.n_step_transition(n)

    def transition_graph(self) -> dict[State, tuple[State, ...]]:
        """Return the directed graph induced by positive transitions.
        
        
        Returns
        -------
        dict[State, tuple[State, ...]]
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `transition_graph`."""
        return {state: tuple(self._states[j] for j, p in enumerate(self._P[i]) if p > self._tolerance) for i, state in enumerate(self._states)}

    def accessible(self, source: State, target: State) -> bool:
        """Return whether target is accessible from source.
        
        Parameters
        ----------
        source : State
            Source state label.
        target : State
            Target state label.
        
        Returns
        -------
        bool
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `accessible`."""
        start, goal = self._idx(source), self._idx(target)
        seen, stack = {start}, [start]
        while stack:
            i = stack.pop()
            for j, p in enumerate(self._P[i]):
                if p > self._tolerance and j not in seen:
                    seen.add(j)
                    stack.append(j)
        return goal in seen

    def communicate(self, source: State, target: State) -> bool:
        """Return whether source and target communicate.
        
        Parameters
        ----------
        source : State
            Source state label.
        target : State
            Target state label.
        
        Returns
        -------
        bool
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `communicate`."""
        return self.accessible(source, target) and self.accessible(target, source)

    def communicating_classes(self) -> list[tuple[State, ...]]:
        """Return the communicating classes of the transition graph.
        
        
        Returns
        -------
        list[tuple[State, ...]]
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `communicating_classes`."""
        graph = [[j for j, p in enumerate(row) if p > self._tolerance] for row in self._P]
        reverse = [[] for _ in graph]
        for i, neighbours in enumerate(graph):
            for j in neighbours:
                reverse[j].append(i)
        seen = [False] * self.n_states
        order: list[int] = []

        def dfs(v: int) -> None:
            """Public stochastic operation.
            
            Parameters
            ----------
            v : int
                Input argument.
            
            Examples
            --------
            See the executable examples and API reference for `dfs`."""
            seen[v] = True
            for w in graph[v]:
                if not seen[w]:
                    dfs(w)
            order.append(v)

        for v in range(self.n_states):
            if not seen[v]:
                dfs(v)
        seen = [False] * self.n_states
        classes: list[tuple[State, ...]] = []

        def rdfs(v: int, component: list[int]) -> None:
            """Public stochastic operation.
            
            Parameters
            ----------
            v : int
                Input argument.
            component : list[int]
                Input argument.
            
            Examples
            --------
            See the executable examples and API reference for `rdfs`."""
            seen[v] = True
            component.append(v)
            for w in reverse[v]:
                if not seen[w]:
                    rdfs(w, component)

        for v in reversed(order):
            if not seen[v]:
                component: list[int] = []
                rdfs(v, component)
                classes.append(tuple(self._states[i] for i in component))
        return classes

    def is_irreducible(self) -> bool:
        """Return whether all states communicate.
        
        
        Returns
        -------
        bool
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `is_irreducible`."""
        return len(self.communicating_classes()) == 1

    def closed_classes(self) -> list[tuple[State, ...]]:
        """Return communicating classes from which no transition leaves.
        
        
        Returns
        -------
        list[tuple[State, ...]]
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `closed_classes`."""
        graph = self.transition_graph()
        result: list[tuple[State, ...]] = []
        for component in self.communicating_classes():
            members = set(component)
            if all(set(graph[state]).issubset(members) for state in component):
                result.append(component)
        return result

    def is_absorbing_state(self, state: State) -> bool:
        """Return whether ``p_ii=1``.
        
        Parameters
        ----------
        state : State
            State label.
        
        Returns
        -------
        bool
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `is_absorbing_state`."""
        i = self._idx(state)
        return abs(self._P[i, i] - 1.0) <= self._tolerance

    def classify_states(self) -> dict[State, str]:
        """Classify finite-chain states as recurrent or transient.
        
        
        Returns
        -------
        dict[State, str]
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `classify_states`."""
        graph = [{j for j, p in enumerate(row) if p > self._tolerance} for row in self._P]
        classification: dict[State, str] = {}
        for component in self.communicating_classes():
            ids = {self._idx(s) for s in component}
            kind = "recurrent" if all(graph[i].issubset(ids) for i in ids) else "transient"
            for state in component:
                classification[state] = kind
        return classification

    def first_visit_probability(self, source: State, target: State, n: int) -> float:
        """Return ``P_i(T_j=n)``.
        
        Parameters
        ----------
        source : State
            Source state label.
        target : State
            Target state label.
        n : int
            Integer index required by the operation.
        
        Returns
        -------
        float
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `first_visit_probability`."""
        self._positive_int(n, "n")
        i, j = self._idx(source), self._idx(target)
        if i == j:
            return 0.0
        killed = self._P.copy()
        killed[:, j] = 0.0
        row = np.zeros(self.n_states)
        row[i] = 1.0
        if n > 1:
            row = row @ np.linalg.matrix_power(killed, n - 1)
        return float(row @ self._P[:, j])

    def first_passage_probability(self, source: State, target: State, n: int) -> float:
        """Return the canonical first-passage probability ``P_i(T_j=n)``.
        
        Parameters
        ----------
        source : State
            Source state label.
        target : State
            Target state label.
        n : int
            Integer index required by the operation.
        
        Returns
        -------
        float
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `first_passage_probability`."""
        return self.first_visit_probability(source, target, n)

    def visit_probability(self, source: State, target: State, n: int) -> float:
        """Return ``sum_{k=1}^n P_i(T_j=k)``.
        
        Parameters
        ----------
        source : State
            Source state label.
        target : State
            Target state label.
        n : int
            Integer index required by the operation.
        
        Returns
        -------
        float
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `visit_probability`."""
        self._positive_int(n, "n")
        return float(sum(self.first_visit_probability(source, target, k) for k in range(1, n + 1)))

    def hitting_probability(self, source: State, target: State) -> float:
        """Return the probability of ever reaching target from source.
        
        Parameters
        ----------
        source : State
            Source state label.
        target : State
            Target state label.
        
        Returns
        -------
        float
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `hitting_probability`."""
        i, j = self._idx(source), self._idx(target)
        if i == j:
            return 1.0
        ids = [k for k in range(self.n_states) if k != j]
        matrix = np.eye(len(ids)) - self._P[np.ix_(ids, ids)]
        rhs = self._P[np.ix_(ids, [j])].ravel()
        try:
            solution = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            return 0.0
        return float(solution[ids.index(i)])

    def expected_hitting_time(self, source: State, target: State) -> float:
        """Return the expected hitting time of target from source.
        
        Parameters
        ----------
        source : State
            Source state label.
        target : State
            Target state label.
        
        Returns
        -------
        float
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `expected_hitting_time`."""
        i, j = self._idx(source), self._idx(target)
        if i == j:
            return 0.0
        if self.hitting_probability(source, target) < 1.0 - 1e-10:
            return float("inf")
        ids = [k for k in range(self.n_states) if k != j]
        solution = np.linalg.solve(np.eye(len(ids)) - self._P[np.ix_(ids, ids)], np.ones(len(ids)))
        return float(solution[ids.index(i)])

    def mean_hitting_time(self, source: State, target: State) -> float:
        """Return the canonical mean first-hitting time.
        
        Parameters
        ----------
        source : State
            Source state label.
        target : State
            Target state label.
        
        Returns
        -------
        float
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `mean_hitting_time`."""
        return self.expected_hitting_time(source, target)

    def first_return_probability(self, state: State, n: int) -> float:
        """Return the first-return probability ``P_i(T_i=n)``.
        
        Parameters
        ----------
        state : State
            State label.
        n : int
            Integer index required by the operation.
        
        Returns
        -------
        float
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `first_return_probability`."""
        self._positive_int(n, "n")
        i = self._idx(state)
        killed = self._P.copy()
        killed[:, i] = 0.0
        row = np.zeros(self.n_states)
        row[i] = 1.0
        if n > 1:
            row = row @ np.linalg.matrix_power(killed, int(n - 1))
        return float(row @ self._P[:, i])

    def return_probability(self, state: State) -> float:
        """Return ``P_i(T_i<infinity)`` from recurrence classification.
        
        Parameters
        ----------
        state : State
            State label.
        
        Returns
        -------
        float
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `return_probability`."""
        return 1.0 if self.classify_states()[state] == "recurrent" else 0.0

    def mean_return_time(self, state: State) -> float:
        """Return the mean return time ``mu_i`` in a recurrent class.
        
        Parameters
        ----------
        state : State
            State label.
        
        Returns
        -------
        float
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `mean_return_time`."""
        if self.classify_states()[state] != "recurrent":
            return float("inf")
        component = next(c for c in self.communicating_classes() if state in c)
        ids = [self._idx(s) for s in component]
        local = self._stationary(self._P[np.ix_(ids, ids)])
        return float(1.0 / local[ids.index(self._idx(state))])

    def period(self, state: State) -> int | float:
        """Return the period of a state.
        
        Parameters
        ----------
        state : State
            State label.
        
        Returns
        -------
        int | float
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `period`."""
        root = self._idx(state)
        graph = [[j for j, p in enumerate(row) if p > self._tolerance] for row in self._P]
        distances = {root: 0}
        queue = [root]
        d = 0
        while queue:
            vertex = queue.pop(0)
            for neighbour in graph[vertex]:
                if neighbour not in distances:
                    distances[neighbour] = distances[vertex] + 1
                    queue.append(neighbour)
                if neighbour == root:
                    d = gcd(d, distances[vertex] + 1 - distances[neighbour])
        return float("inf") if d == 0 else abs(d)

    def is_aperiodic(self) -> bool:
        """Return whether every state has period one.
        
        
        Returns
        -------
        bool
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `is_aperiodic`."""
        return all(self.period(state) == 1 for state in self._states)

    def is_ergodic(self) -> bool:
        """Return whether the finite chain is recurrent and aperiodic.
        
        
        Returns
        -------
        bool
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `is_ergodic`."""
        classification = self.classify_states()
        return all(classification[s] == "recurrent" and self.period(s) == 1 for s in self._states)

    def stationary_distribution(self) -> np.ndarray:
        """Return the unique stationary distribution for an irreducible finite chain.
        
        
        Returns
        -------
        np.ndarray
            Result produced by the operation.
        
        Raises
        ------
        ValueError
            Raised when an input or mathematical precondition is violated.
        
        Examples
        --------
        See the executable examples and API reference for `stationary_distribution`."""
        if not self.is_irreducible():
            raise ValueError("stationary_distribution() requires an irreducible finite chain")
        return self._stationary(self._P)

    def stationary_distributions(self) -> tuple[np.ndarray, ...]:
        """Return one stationary law for each closed recurrent class.
        
        
        Returns
        -------
        tuple[np.ndarray, ...]
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `stationary_distributions`."""
        vectors = []
        for component in self.closed_classes():
            ids = [self._idx(s) for s in component]
            local = self._stationary(self._P[np.ix_(ids, ids)])
            full = np.zeros(self.n_states)
            full[ids] = local
            vectors.append(full)
        return tuple(vectors)

    def limiting_distribution(self) -> np.ndarray:
        """Return the transition-matrix limit in the finite course cases.
        
        
        Returns
        -------
        np.ndarray
            Result produced by the operation.
        
        Raises
        ------
        ValueError
            Raised when an input or mathematical precondition is violated.
        
        Examples
        --------
        See the executable examples and API reference for `limiting_distribution`."""
        if self.is_ergodic():
            return np.tile(self.stationary_distribution(), (self.n_states, 1))
        closed = self.closed_classes()
        if len(closed) != 1:
            raise ValueError("no limiting distribution under the finite-chain course conditions")
        classification = self.classify_states()
        if any(classification[s] != "transient" for s in self._states if s not in closed[0]):
            raise ValueError("no limiting distribution under the finite-chain course conditions")
        ids = [self._idx(s) for s in closed[0]]
        submatrix = self._P[np.ix_(ids, ids)]
        if not self._matrix_ergodic(submatrix):
            raise ValueError("no limiting distribution under the finite-chain course conditions")
        result = np.zeros((self.n_states, self.n_states))
        result[:, ids] = self._stationary(submatrix)
        return result

    def absorption_probability(self, source: State, absorbing_class: Sequence[State]) -> float:
        """Return the probability of eventual absorption in a closed class.
        
        Parameters
        ----------
        source : State
            Source state label.
        absorbing_class : Sequence[State]
            Input argument.
        
        Returns
        -------
        float
            Result produced by the operation.
        
        Raises
        ------
        ValueError
            Raised when an input or mathematical precondition is violated.
        
        Examples
        --------
        See the executable examples and API reference for `absorption_probability`."""
        target = set(absorbing_class)
        if not target or not any(set(component) == target for component in self.closed_classes()):
            raise ValueError("absorbing_class must be a closed communicating class")
        source_index = self._idx(source)
        target_ids = {self._idx(state) for state in target}
        if source_index in target_ids:
            return 1.0
        if self.classify_states()[self._states[source_index]] != "transient":
            return 0.0
        transient_ids = [k for k, state in enumerate(self._states) if self.classify_states()[state] == "transient"]
        matrix = np.eye(len(transient_ids)) - self._P[np.ix_(transient_ids, transient_ids)]
        rhs = self._P[np.ix_(transient_ids, sorted(target_ids))].sum(axis=1)
        solution = np.linalg.solve(matrix, rhs)
        return float(solution[transient_ids.index(source_index)])

    def simulate(self, n_steps: int, *, initial_state: State | None = None, initial_distribution: Sequence[float] | None = None, rng: np.random.Generator | None = None) -> list[State]:
        """Simulate a path ``X_0,...,X_n`` from the transition matrix.
        
        Parameters
        ----------
        n_steps : int
            Number of discrete transitions.
        initial_state : State | None
            Optional initial state.
        initial_distribution : Sequence[float] | None
            Initial probability distribution.
        rng : np.random.Generator | None
            Optional NumPy random generator.
        
        Returns
        -------
        list[State]
            Result produced by the operation.
        
        Raises
        ------
        ValueError
            Raised when an input or mathematical precondition is violated.
        
        Examples
        --------
        See the executable examples and API reference for `simulate`."""
        self._nonnegative_int(n_steps, "n_steps")
        if initial_state is not None and initial_distribution is not None:
            raise ValueError("provide only one initial condition")
        generator = rng if rng is not None else np.random.default_rng()
        if initial_distribution is not None:
            current = int(generator.choice(self.n_states, p=self._distribution(initial_distribution)))
        elif initial_state is None:
            current = 0
        else:
            current = self._idx(initial_state)
        path = [self._states[current]]
        for _ in range(int(n_steps)):
            current = int(generator.choice(self.n_states, p=self._P[current]))
            path.append(self._states[current])
        return path

    def jump_chain(self) -> "MarkovChain":
        """Return this chain; a discrete-time chain is already an embedded chain.
        
        
        Returns
        -------
        'MarkovChain'
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `jump_chain`."""
        return self

    def _idx(self, state: State) -> int:
        try:
            return self._index[state]
        except KeyError as exc:
            raise ValueError(f"unknown state: {state!r}") from exc

    def _distribution(self, values: Sequence[float]) -> np.ndarray:
        distribution = np.asarray(values, dtype=float)
        if distribution.shape != (self.n_states,) or not np.all(np.isfinite(distribution)) or np.any(distribution < 0) or not np.isclose(distribution.sum(), 1.0):
            raise ValueError("distribution must be non-negative and sum to 1")
        return distribution

    @staticmethod
    def _nonnegative_int(value: int, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    @staticmethod
    def _positive_int(value: int, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    @staticmethod
    def _stationary(matrix: np.ndarray) -> np.ndarray:
        n = matrix.shape[0]
        system = matrix.T - np.eye(n)
        system[-1] = 1.0
        rhs = np.zeros(n)
        rhs[-1] = 1.0
        distribution = np.linalg.solve(system, rhs)
        distribution[np.abs(distribution) < 1e-12] = 0.0
        return distribution / distribution.sum()

    @staticmethod
    def _matrix_ergodic(matrix: np.ndarray) -> bool:
        """Check irreducibility and period one for a finite stochastic matrix."""
        n = matrix.shape[0]
        if n == 1:
            return True
        graph = [[j for j, p in enumerate(row) if p > 1e-12] for row in matrix]
        for start in range(n):
            seen = {start}
            queue = [start]
            while queue:
                vertex = queue.pop()
                for neighbour in graph[vertex]:
                    if neighbour not in seen:
                        seen.add(neighbour)
                        queue.append(neighbour)
            if len(seen) != n:
                return False
        period = 0
        distances = {0: 0}
        queue = [0]
        while queue:
            vertex = queue.pop(0)
            for neighbour in graph[vertex]:
                if neighbour not in distances:
                    distances[neighbour] = distances[vertex] + 1
                    queue.append(neighbour)
                if neighbour == 0:
                    period = gcd(period, distances[vertex] + 1 - distances[neighbour])
        return period == 1


__all__ = ["MarkovChain"]
