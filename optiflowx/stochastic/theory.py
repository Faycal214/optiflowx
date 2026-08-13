"""Mathematical consequences used in the five MSPRO Processus Aléatoires PDFs.

This module contains small helpers for results that sit above the core
probability/process objects.  It deliberately stays within the five supplied
course chapters and does not introduce an additional stochastic framework.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from math import gcd
from typing import Hashable

import numpy as np

from .cmtc import CTMCPath, ContinuousTimeMarkovChain
from .conditional import FiniteProbabilitySpace, Partition, RandomVariable
from .markov import MarkovChain
from .martingale import Martingale, StoppingTime

State = Hashable


def first_return_probability(chain: MarkovChain, state: State, n: int) -> float:
    """Return P_i(T_i=n), the first-return probability at step n."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    i = _state_index(chain.states, state)
    killed = chain.transition_matrix.copy()
    killed[:, i] = 0.0
    row = np.zeros(chain.n_states)
    row[i] = 1.0
    if n > 1:
        row = row @ np.linalg.matrix_power(killed, int(n - 1))
    return float(row @ chain.transition_matrix[:, i])


def return_probability(chain: MarkovChain, state: State) -> float:
    """Return P_i(T_i < infinity) using the course's recurrence classification."""
    classification = chain.classify_states()[state]
    return 1.0 if classification == "recurrent" else 0.0


def mean_return_time(chain: MarkovChain, state: State) -> float:
    """Return the mean return time mu_i for a finite positive recurrent class."""
    if chain.classify_states()[state] != "recurrent":
        return float("inf")
    component = next(c for c in chain.communicating_classes() if state in c)
    ids = [_state_index(chain.states, s) for s in component]
    sub = chain.transition_matrix[np.ix_(ids, ids)]
    pi = _stationary_distribution(sub)
    return float(1.0 / pi[ids.index(_state_index(chain.states, state))])


def stationary_distributions(chain: MarkovChain) -> tuple[np.ndarray, ...]:
    """Return one stationary probability vector for each closed recurrent class.

    Any convex combination of these vectors is also stationary.  For an
    irreducible finite chain the tuple contains the unique stationary law.
    """
    vectors: list[np.ndarray] = []
    for component in chain.closed_classes():
        ids = [_state_index(chain.states, s) for s in component]
        sub = chain.transition_matrix[np.ix_(ids, ids)]
        local = _stationary_distribution(sub)
        full = np.zeros(chain.n_states)
        full[ids] = local
        vectors.append(full)
    return tuple(vectors)


def empirical_state_frequencies(path: Sequence[State], states: Sequence[State]) -> np.ndarray:
    """Return empirical visit frequencies of states along a discrete trajectory."""
    labels = tuple(states)
    if not path:
        raise ValueError("path must not be empty")
    index = {state: i for i, state in enumerate(labels)}
    if len(index) != len(labels):
        raise ValueError("states must be unique")
    counts = np.zeros(len(labels), dtype=float)
    for state in path:
        if state not in index:
            raise ValueError(f"unknown state in path: {state!r}")
        counts[index[state]] += 1.0
    return counts / len(path)


def ctmc_communication_classes(chain: ContinuousTimeMarkovChain) -> list[tuple[State, ...]]:
    """Return CTMC communication classes via the embedded jump chain."""
    return chain.jump_chain().communicating_classes()


def ctmc_stationary_from_jump_chain(chain: ContinuousTimeMarkovChain) -> np.ndarray:
    """Recover the CTMC stationary law from the stationary law of its jump chain.

    For q_i=-q_ii>0 and jump-chain stationary law phi,
    pi_i = (phi_i/q_i) / sum_j(phi_j/q_j).
    """
    rates = np.asarray([-chain.generator_matrix[i, i] for i in range(chain.n_states)], dtype=float)
    if np.any(rates <= 0):
        raise ValueError("the course relation requires positive holding rates in every state")
    phi = chain.jump_chain().stationary_distribution()
    weights = phi / rates
    return weights / weights.sum()


def ctmc_mean_return_time(chain: ContinuousTimeMarkovChain, state: State) -> float:
    """Return the mean continuous-time return time mu_i."""
    pi = chain.stationary_distribution()
    i = _state_index(chain.states, state)
    rate = -chain.generator_matrix[i, i]
    if rate <= 0 or pi[i] <= 0:
        return float("inf")
    return float(1.0 / (rate * pi[i]))


def occupation_time(path: CTMCPath, state: State, horizon: float) -> float:
    """Return time spent in ``state`` over [0, horizon]."""
    if not np.isfinite(horizon) or horizon < 0:
        raise ValueError("horizon must be finite and non-negative")
    if horizon == 0:
        return 0.0
    if len(path.times) == 0 or path.times[0] != 0.0:
        raise ValueError("CTMC path must start at time 0")
    total = 0.0
    times = np.asarray(path.times, dtype=float)
    for i, start in enumerate(times):
        if start >= horizon:
            break
        end = horizon if i + 1 == len(times) else min(float(times[i + 1]), horizon)
        if path.states[i] == state:
            total += max(0.0, end - float(start))
    return float(total)


def occupation_fraction(path: CTMCPath, state: State, horizon: float) -> float:
    """Return the empirical occupation fraction over [0, horizon]."""
    if horizon <= 0 or not np.isfinite(horizon):
        raise ValueError("horizon must be strictly positive and finite")
    return occupation_time(path, state, horizon) / float(horizon)


def conditional_expectation_given_event(
    space: FiniteProbabilitySpace,
    x: RandomVariable,
    event: Iterable[Hashable],
) -> float:
    """Compute E[X | B] for an event B of positive probability."""
    if x.space is not space:
        raise ValueError("random variable belongs to another probability space")
    event_set = set(event)
    p = space.probability(event_set)
    if p <= 0:
        raise ValueError("conditioning event must have positive probability")
    return float(sum(space.probabilities[o] * x.values[o] for o in event_set) / p)


def conditional_probability_given_event(
    space: FiniteProbabilitySpace,
    event: Iterable[Hashable],
    conditioning_event: Iterable[Hashable],
) -> float:
    """Compute P(A | B) for positive-probability B."""
    a = set(event)
    b = set(conditioning_event)
    p_b = space.probability(b)
    if p_b <= 0:
        raise ValueError("conditioning event must have positive probability")
    return float(space.probability(a & b) / p_b)


def independent_partitions(
    space: FiniteProbabilitySpace,
    first: Partition,
    second: Partition,
    *,
    atol: float = 1e-12,
) -> bool:
    """Check independence of two finite partitions by block intersections."""
    for a in first.blocks:
        for b in second.blocks:
            if not np.isclose(space.probability(a & b), space.probability(a) * space.probability(b), atol=atol, rtol=0.0):
                return False
    return True


def independent_random_variables(
    space: FiniteProbabilitySpace,
    first: RandomVariable,
    second: RandomVariable,
    *,
    atol: float = 1e-12,
) -> bool:
    """Check independence through the sigma-fields generated by two discrete variables."""
    if first.space is not space or second.space is not space:
        raise ValueError("random variables must belong to the supplied probability space")
    return independent_partitions(
        space,
        Partition.generated_by(first),
        Partition.generated_by(second),
        atol=atol,
    )


def conditional_characterization_error(
    space: FiniteProbabilitySpace,
    x: RandomVariable,
    partition: Partition,
) -> float:
    """Maximum error in E[1_A X] = E[1_A E(X|G)] over G-blocks."""
    ce = space.conditional_expectation(x, partition)
    error = 0.0
    for block in partition.blocks:
        indicator = space.random_variable(
            [float(o in block) for o in space.outcomes],
            name="1_A",
        )
        error = max(error, abs((indicator * x).expectation() - (indicator * ce).expectation()))
    return float(error)


def transform_martingale(
    martingale: Martingale,
    function: Callable[[float], float],
    *,
    name: str | None = None,
) -> Martingale:
    """Apply a scalar transformation to every variable of a martingale process."""
    transformed = tuple(rv.apply(function, name=name) for rv in martingale.process)
    return Martingale(transformed, martingale.filtration)


def stopped_martingale(
    martingale: Martingale,
    stopping_time: StoppingTime,
) -> Martingale:
    """Return the stopped process as a martingale on the same supplied filtration."""
    stopped = martingale.stopped(stopping_time).sequence()
    return Martingale(stopped, martingale.filtration)


def _state_index(states: Sequence[State], state: State) -> int:
    try:
        return tuple(states).index(state)
    except ValueError as exc:
        raise ValueError(f"unknown state: {state!r}") from exc


def _stationary_distribution(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    if n == 0:
        raise ValueError("matrix must be non-empty")
    a = matrix.T - np.eye(n)
    a[-1] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    pi = np.linalg.solve(a, b)
    if np.any(pi < -1e-10):
        raise ValueError("no non-negative stationary distribution")
    pi[np.abs(pi) < 1e-12] = 0.0
    return pi / pi.sum()
