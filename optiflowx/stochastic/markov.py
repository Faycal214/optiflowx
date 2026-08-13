"""Finite-state, homogeneous discrete-time Markov chains (CMTD)."""

from __future__ import annotations

from collections.abc import Sequence
from math import gcd
from typing import Hashable

import numpy as np

State = Hashable


class MarkovChain:
    """Finite-state homogeneous discrete-time Markov chain."""

    def __init__(self, transition_matrix: Sequence[Sequence[float]], states: Sequence[State] | None = None, *, tolerance: float = 1e-12) -> None:
        matrix = np.asarray(transition_matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
            raise ValueError("transition_matrix must be a non-empty square matrix")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("transition_matrix must contain finite values")
        if np.any(matrix < -tolerance) or not np.allclose(matrix.sum(axis=1), 1.0, atol=tolerance, rtol=0.0):
            raise ValueError("transition_matrix must be stochastic: non-negative rows summing to 1")
        matrix = matrix.copy(); matrix[np.abs(matrix) < tolerance] = 0.0; matrix[matrix < 0.0] = 0.0
        labels = tuple(range(matrix.shape[0])) if states is None else tuple(states)
        if len(labels) != matrix.shape[0] or len(set(labels)) != len(labels):
            raise ValueError("states must be unique and match matrix size")
        self._P, self._states, self._index, self._tolerance = matrix, labels, {s: i for i, s in enumerate(labels)}, tolerance

    @property
    def transition_matrix(self) -> np.ndarray: return self._P.copy()
    @property
    def states(self) -> tuple[State, ...]: return self._states
    @property
    def n_states(self) -> int: return len(self._states)

    def n_step_transition(self, n: int) -> np.ndarray:
        self._nonnegative_int(n, "n"); return np.linalg.matrix_power(self._P, int(n))

    def state_distribution(self, initial_distribution: Sequence[float], n: int) -> np.ndarray:
        self._nonnegative_int(n, "n"); return self._distribution(initial_distribution) @ self.n_step_transition(n)

    def chapman_kolmogorov(self, m: int, n: int) -> np.ndarray:
        self._nonnegative_int(m, "m"); self._nonnegative_int(n, "n"); return self.n_step_transition(m) @ self.n_step_transition(n)

    def transition_graph(self) -> dict[State, tuple[State, ...]]:
        return {s: tuple(self._states[j] for j, p in enumerate(self._P[i]) if p > self._tolerance) for i, s in enumerate(self._states)}

    def accessible(self, source: State, target: State) -> bool:
        start, goal = self._idx(source), self._idx(target); seen, stack = {start}, [start]
        while stack:
            i = stack.pop()
            for j, p in enumerate(self._P[i]):
                if p > self._tolerance and j not in seen: seen.add(j); stack.append(j)
        return goal in seen

    def communicate(self, source: State, target: State) -> bool: return self.accessible(source, target) and self.accessible(target, source)

    def communicating_classes(self) -> list[tuple[State, ...]]:
        graph = [[j for j, p in enumerate(row) if p > self._tolerance] for row in self._P]; rev = [[] for _ in graph]
        for i, ns in enumerate(graph):
            for j in ns: rev[j].append(i)
        seen, order = [False]*self.n_states, []
        def dfs(i):
            seen[i] = True
            for j in graph[i]:
                if not seen[j]: dfs(j)
            order.append(i)
        for i in range(self.n_states):
            if not seen[i]: dfs(i)
        seen, out = [False]*self.n_states, []
        def rdfs(i, comp):
            seen[i] = True; comp.append(i)
            for j in rev[i]:
                if not seen[j]: rdfs(j, comp)
        for i in reversed(order):
            if not seen[i]:
                comp=[]; rdfs(i, comp); out.append(tuple(self._states[j] for j in comp))
        return out

    def is_irreducible(self) -> bool: return len(self.communicating_classes()) == 1

    def closed_classes(self) -> list[tuple[State, ...]]:
        graph = self.transition_graph(); out=[]
        for comp in self.communicating_classes():
            members=set(comp)
            if all(set(graph[s]).issubset(members) for s in comp): out.append(comp)
        return out

    def is_absorbing_state(self, state: State) -> bool: return abs(self._P[self._idx(state), self._idx(state)] - 1.0) <= self._tolerance

    def classify_states(self) -> dict[State, str]:
        graph=[{j for j,p in enumerate(row) if p > self._tolerance} for row in self._P]; out={}
        for comp in self.communicating_classes():
            ids={self._idx(s) for s in comp}; label="recurrent" if all(graph[i].issubset(ids) for i in ids) else "transient"
            for s in comp: out[s]=label
        return out

    def period(self, state: State) -> int | float:
        root=self._idx(state); graph=[[j for j,p in enumerate(row) if p > self._tolerance] for row in self._P]; dist={root:0}; q=[root]; d=0
        while q:
            u=q.pop(0)
            for v in graph[u]:
                if v not in dist: dist[v]=dist[u]+1; q.append(v)
                if v==root: d=gcd(d, dist[u]+1-dist[v])
        return float("inf") if d==0 else abs(d)

    def is_aperiodic(self) -> bool: return all(self.period(s)==1 for s in self._states)
    def is_ergodic(self) -> bool:
        cls=self.classify_states(); return all(cls[s]=="recurrent" and self.period(s)==1 for s in self._states)

    def first_visit_probability(self, source: State, target: State, n: int) -> float:
        self._positive_int(n, "n"); i,j=self._idx(source),self._idx(target)
        killed=self._P.copy(); killed[j,:]=0.0; row=np.zeros(self.n_states); row[i]=1.0
        if n>1: row=row @ np.linalg.matrix_power(killed,n-1)
        return float(row @ self._P[:,j])

    def visit_probability(self, source: State, target: State, n: int) -> float:
        self._positive_int(n, "n"); return float(sum(self.first_visit_probability(source,target,k) for k in range(1,n+1)))

    def hitting_probability(self, source: State, target: State) -> float:
        i,j=self._idx(source),self._idx(target)
        if i==j: return 1.0
        ids=[k for k in range(self.n_states) if k!=j]; A=np.eye(len(ids))-self._P[np.ix_(ids,ids)]; b=self._P[np.ix_(ids,[j])].ravel()
        try: return float(np.linalg.solve(A,b)[ids.index(i)])
        except np.linalg.LinAlgError: return 0.0

    def expected_hitting_time(self, source: State, target: State) -> float:
        i,j=self._idx(source),self._idx(target)
        if i==j: return 0.0
        if self.hitting_probability(source,target) < 1.0-1e-10: return float("inf")
        ids=[k for k in range(self.n_states) if k!=j]; A=np.eye(len(ids))-self._P[np.ix_(ids,ids)]
        return float(np.linalg.solve(A,np.ones(len(ids)))[ids.index(i)])

    def stationary_distribution(self) -> np.ndarray:
        if not self.is_irreducible(): raise ValueError("stationary_distribution() requires an irreducible finite chain")
        return self._stationary(self._P)

    def limiting_distribution(self) -> np.ndarray:
        if self.is_ergodic(): return np.tile(self.stationary_distribution(),(self.n_states,1))
        closed=self.closed_classes()
        if len(closed)!=1: raise ValueError("the course conditions for a limiting distribution are not met")
        cls=self.classify_states()
        if any(cls[s]!="transient" for s in self._states if s not in closed[0]): raise ValueError("the unique closed class is not the only recurrent class")
        ids=[self._idx(s) for s in closed[0]]; sub=self._P[np.ix_(ids,ids)]
        if not self._matrix_ergodic(sub): raise ValueError("the unique closed class is not ergodic")
        out=np.zeros((self.n_states,self.n_states)); out[:,ids]=self._stationary(sub); return out

    def absorption_probability(self, source: State, absorbing_class: Sequence[State]) -> float:
        target=set(absorbing_class)
        if not target or not any(set(c)==target for c in self.closed_classes()): raise ValueError("absorbing_class must be a closed communicating class")
        i=self._idx(source); tids={self._idx(s) for s in target}
        if i in tids: return 1.0
        ids=[k for k in range(self.n_states) if k not in tids]; A=np.eye(len(ids))-self._P[np.ix_(ids,ids)]; b=self._P[np.ix_(ids,sorted(tids))].sum(axis=1)
        return float(np.linalg.solve(A,b)[ids.index(i)])

    def simulate(self,n_steps:int,*,initial_state:State|None=None,initial_distribution:Sequence[float]|None=None,rng:np.random.Generator|None=None)->list[State]:
        self._nonnegative_int(n_steps,"n_steps")
        if initial_state is not None and initial_distribution is not None: raise ValueError("provide only one initial condition")
        g=rng if rng is not None else np.random.default_rng()
        if initial_distribution is not None: cur=int(g.choice(self.n_states,p=self._distribution(initial_distribution)))
        elif initial_state is None: cur=0
        else: cur=self._idx(initial_state)
        path=[self._states[cur]]
        for _ in range(int(n_steps)): cur=int(g.choice(self.n_states,p=self._P[cur])); path.append(self._states[cur])
        return path

    def _idx(self,s):
        try: return self._index[s]
        except KeyError as exc: raise ValueError(f"unknown state: {s!r}") from exc
    def _distribution(self,x,name="distribution"):
        mu=np.asarray(x,dtype=float)
        if mu.shape!=(self.n_states,) or not np.all(np.isfinite(mu)) or np.any(mu<0) or not np.isclose(mu.sum(),1.0): raise ValueError(f"{name} must be non-negative and sum to 1")
        return mu
    @staticmethod
    def _nonnegative_int(x,name):
        if isinstance(x,bool) or not isinstance(x,(int,np.integer)) or x<0: raise ValueError(f"{name} must be a non-negative integer")
    @staticmethod
    def _positive_int(x,name):
        if isinstance(x,bool) or not isinstance(x,(int,np.integer)) or x<1: raise ValueError(f"{name} must be a positive integer")
    @staticmethod
    def _stationary(m):
        n=m.shape[0]; A=m.T-np.eye(n); A[-1]=1.0; b=np.zeros(n); b[-1]=1.0; p=np.linalg.solve(A,b); p[np.abs(p)<1e-12]=0.0; return p/p.sum()
    @staticmethod
    def _matrix_ergodic(m):
        n=m.shape[0]
        if n==1: return True
        graph=[[j for j,p in enumerate(row) if p>1e-12] for row in m]
        for start in range(n):
            seen={start}; q=[start]
            while q:
                u=q.pop()
                for v in graph[u]:
                    if v not in seen: seen.add(v); q.append(v)
            if len(seen)!=n: return False
        d=0; start=0; dist={start:0}; q=[start]
        while q:
            u=q.pop(0)
            for v in graph[u]:
                if v not in dist: dist[v]=dist[u]+1; q.append(v)
                if v==start: d=gcd(d,dist[u]+1-dist[v])
        return d==1
