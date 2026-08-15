"""Finite probability spaces and conditional expectation from MSPRO Chapter 4."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Hashable

import numpy as np

from .partition import Partition
from .random_variable import RandomVariable

Outcome = Hashable


class FiniteProbabilitySpace:
    """Finite probability space used for the chapter's discrete framework.
    
    Mathematical object
    ------------------
    Public stochastic object exposed by the OptiFlowX API.
    
    Course basis
    ------------
    The implementation follows the corresponding MSPRO course material documented by OptiFlowX.
    
    Parameters
    ----------
    outcomes : Sequence[Outcome]
        Finite outcome labels.
    probabilities : Sequence[float]
        Probability vector.
    
    Examples
    --------
    See the executable examples for `probability_space.py` and the API reference."""

    def __init__(self, outcomes: Sequence[Outcome], probabilities: Sequence[float]) -> None:
        """Create ``(Omega, P)`` from its outcomes and point probabilities.
        
        Parameters
        ----------
        outcomes : Sequence[Outcome]
            Finite outcome labels.
        probabilities : Sequence[float]
            Probability vector.
        
        Raises
        ------
        ValueError
            Raised when an input or mathematical precondition is violated.
        
        Examples
        --------
        See the executable examples and API reference for `__init__`."""
        self.outcomes = tuple(outcomes)
        if not self.outcomes or len(set(self.outcomes)) != len(self.outcomes):
            raise ValueError("outcomes must be unique and non-empty")
        p = np.asarray(probabilities, dtype=float)
        if p.shape != (len(self.outcomes),) or not np.all(np.isfinite(p)) or np.any(p < 0) or not np.isclose(p.sum(), 1.0):
            raise ValueError("probabilities must be non-negative and sum to 1")
        self.probabilities = dict(zip(self.outcomes, p.astype(float)))

    @property
    def n_outcomes(self) -> int:
        """Return ``|Omega|``.
        
        
        Returns
        -------
        int
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `n_outcomes`."""
        return len(self.outcomes)

    @property
    def probabilities_array(self) -> np.ndarray:
        """Return point probabilities in outcome order.
        
        
        Returns
        -------
        np.ndarray
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `probabilities_array`."""
        return np.asarray([self.probabilities[o] for o in self.outcomes])

    def random_variable(self, values: Mapping[Outcome, float] | Sequence[float], *, name: str | None = None) -> RandomVariable:
        """Create a discrete random variable on this space.
        
        Parameters
        ----------
        values : Mapping[Outcome, float] | Sequence[float]
            Values or mapping.
        name : str | None
            Optional display name.
        
        Returns
        -------
        RandomVariable
            Result produced by the operation.
        
        Raises
        ------
        ValueError
            Raised when an input or mathematical precondition is violated.
        
        Examples
        --------
        See the executable examples and API reference for `random_variable`."""
        if isinstance(values, Mapping):
            mapping = values
        else:
            if len(values) != len(self.outcomes):
                raise ValueError("values must have one entry per outcome")
            mapping = dict(zip(self.outcomes, values))
        return RandomVariable(self, mapping, name=name)

    def partition(self, blocks: Iterable[Iterable[Outcome]]) -> Partition:
        """Create a finite sigma-field representation from partition blocks.
        
        Parameters
        ----------
        blocks : Iterable[Iterable[Outcome]]
            Partition blocks.
        
        Returns
        -------
        Partition
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `partition`."""
        return Partition.from_blocks(blocks, self)

    def probability(self, event: Iterable[Outcome]) -> float:
        """Return the probability of a finite event.
        
        Parameters
        ----------
        event : Iterable[Outcome]
            Finite event.
        
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
        See the executable examples and API reference for `probability`."""
        event = set(event)
        if not event.issubset(self.outcomes):
            raise ValueError("event contains an unknown outcome")
        return float(sum(self.probabilities[o] for o in event))

    def probability_of(self, event: Iterable[Outcome]) -> float:
        """Return ``P(A)`` using the canonical API name.
        
        Parameters
        ----------
        event : Iterable[Outcome]
            Finite event.
        
        Returns
        -------
        float
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `probability_of`."""
        return self.probability(event)

    def conditional_probability_given_event(self, event: Iterable[Outcome], condition: Iterable[Outcome]) -> float:
        """Return ``P(A|B)=P(A intersect B)/P(B)`` when ``P(B)>0``.
        
        Parameters
        ----------
        event : Iterable[Outcome]
            Finite event.
        condition : Iterable[Outcome]
            Conditioning event.
        
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
        See the executable examples and API reference for `conditional_probability_given_event`."""
        a = set(event)
        b = set(condition)
        if not a.issubset(self.outcomes) or not b.issubset(self.outcomes):
            raise ValueError("event contains an unknown outcome")
        denominator = self.probability(b)
        if denominator <= 0:
            raise ValueError("conditioning event must have positive probability")
        return self.probability(a & b) / denominator

    def conditional_expectation_given_event(self, x: RandomVariable, condition: Iterable[Outcome]) -> float:
        """Return ``E(X|B)=E(X 1_B)/P(B)`` for ``P(B)>0``.
        
        Parameters
        ----------
        x : RandomVariable
            Random variable or numeric input.
        condition : Iterable[Outcome]
            Conditioning event.
        
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
        See the executable examples and API reference for `conditional_expectation_given_event`."""
        if x.space is not self:
            raise ValueError("random variable belongs to another probability space")
        b = set(condition)
        denominator = self.probability(b)
        if denominator <= 0:
            raise ValueError("conditioning event must have positive probability")
        numerator = sum(self.probabilities[o] * x.values[o] for o in b)
        return float(numerator / denominator)

    def conditional_expectation(self, x: RandomVariable, partition: Partition) -> RandomVariable:
        """Compute ``E(X | G)`` block-by-block.
        
        Parameters
        ----------
        x : RandomVariable
            Random variable or numeric input.
        partition : Partition
            Conditioning partition.
        
        Returns
        -------
        RandomVariable
            Result produced by the operation.
        
        Raises
        ------
        TypeError
            Raised when an input or mathematical precondition is violated.
        ValueError
            Raised when an input or mathematical precondition is violated.
        
        Examples
        --------
        See the executable examples and API reference for `conditional_expectation`."""
        if x.space is not self:
            raise ValueError("random variable belongs to another probability space")
        if not isinstance(partition, Partition):
            raise TypeError("partition must be a Partition")
        values = {}
        for block in partition.blocks:
            mass = self.probability(block)
            if mass <= 0:
                raise ValueError("conditional expectation is undefined on a zero-probability block")
            mean = sum(self.probabilities[o] * x.values[o] for o in block) / mass
            for o in block:
                values[o] = float(mean)
        return RandomVariable(self, values, name=f"E({x.name or 'X'}|G)")

    def conditional_expectation_given(self, x: RandomVariable, y: RandomVariable) -> RandomVariable:
        """Compute ``E(X | Y)`` through the partition generated by ``Y``.
        
        Parameters
        ----------
        x : RandomVariable
            Random variable or numeric input.
        y : RandomVariable
            Random variable or numeric input.
        
        Returns
        -------
        RandomVariable
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `conditional_expectation_given`."""
        return self.conditional_expectation(x, Partition.generated_by(y))

    def conditional_probability(self, event: Iterable[Outcome], partition: Partition) -> RandomVariable:
        """Return ``P(A|G)`` as ``E(1_A|G)``.
        
        Parameters
        ----------
        event : Iterable[Outcome]
            Finite event.
        partition : Partition
            Conditioning partition.
        
        Returns
        -------
        RandomVariable
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `conditional_probability`."""
        event = set(event)
        indicator = self.random_variable({o: float(o in event) for o in self.outcomes}, name="1_A")
        result = self.conditional_expectation(indicator, partition)
        return RandomVariable(self, result.values, name="P(A|G)")

    def total_expectation(self, x: RandomVariable, partition: Partition) -> float:
        """Return ``E[E(X|G)]``.
        
        Parameters
        ----------
        x : RandomVariable
            Random variable or numeric input.
        partition : Partition
            Conditioning partition.
        
        Returns
        -------
        float
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `total_expectation`."""
        return self.conditional_expectation(x, partition).expectation()

    def tower(self, x: RandomVariable, finer: Partition, coarser: Partition) -> RandomVariable:
        """Apply the tower property when the first partition refines the second.
        
        Parameters
        ----------
        x : RandomVariable
            Random variable or numeric input.
        finer : Partition
            Finer partition.
        coarser : Partition
            Coarser partition.
        
        Returns
        -------
        RandomVariable
            Result produced by the operation.
        
        Raises
        ------
        ValueError
            Raised when an input or mathematical precondition is violated.
        
        Examples
        --------
        See the executable examples and API reference for `tower`."""
        if not finer.refines(coarser):
            raise ValueError("finer partition must refine coarser partition")
        return self.conditional_expectation(self.conditional_expectation(x, finer), coarser)

    def pull_out(self, y: RandomVariable, x: RandomVariable, partition: Partition) -> RandomVariable:
        """Return the residual of ``E(YX|G)=Y E(X|G)``.
        
        Parameters
        ----------
        y : RandomVariable
            Random variable or numeric input.
        x : RandomVariable
            Random variable or numeric input.
        partition : Partition
            Conditioning partition.
        
        Returns
        -------
        RandomVariable
            Result produced by the operation.
        
        Raises
        ------
        ValueError
            Raised when an input or mathematical precondition is violated.
        
        Examples
        --------
        See the executable examples and API reference for `pull_out`."""
        if not self.check_measurable(y, partition):
            raise ValueError("y must be measurable with respect to the conditioning partition")
        left = self.conditional_expectation(y * x, partition)
        right = y * self.conditional_expectation(x, partition)
        return left - right

    def conditional_variance(self, x: RandomVariable, partition: Partition) -> RandomVariable:
        """Return ``Var(X|G)=E(X^2|G)-E(X|G)^2``.
        
        Parameters
        ----------
        x : RandomVariable
            Random variable or numeric input.
        partition : Partition
            Conditioning partition.
        
        Returns
        -------
        RandomVariable
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `conditional_variance`."""
        ex = self.conditional_expectation(x, partition)
        return self.conditional_expectation(x * x, partition) - ex * ex

    def conditional_covariance(self, x: RandomVariable, y: RandomVariable, partition: Partition) -> RandomVariable:
        """Return ``Cov(X,Y|G)``.
        
        Parameters
        ----------
        x : RandomVariable
            Random variable or numeric input.
        y : RandomVariable
            Random variable or numeric input.
        partition : Partition
            Conditioning partition.
        
        Returns
        -------
        RandomVariable
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `conditional_covariance`."""
        return self.conditional_expectation(x * y, partition) - self.conditional_expectation(x, partition) * self.conditional_expectation(y, partition)

    def are_partitions_independent(self, first: Partition, second: Partition, *, atol: float = 1e-12) -> bool:
        """Check independence of two finite partitions.
        
        Parameters
        ----------
        first : Partition
            First partition or random variable.
        second : Partition
            Second partition or random variable.
        atol : float
            Input argument.
        
        Returns
        -------
        bool
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `are_partitions_independent`."""
        for a in first.blocks:
            for b in second.blocks:
                if not np.isclose(self.probability(a & b), self.probability(a) * self.probability(b), atol=atol, rtol=0.0):
                    return False
        return True

    def are_independent(self, first: RandomVariable, second: RandomVariable, *, atol: float = 1e-12) -> bool:
        """Check independence of two finite random variables.
        
        Parameters
        ----------
        first : RandomVariable
            First partition or random variable.
        second : RandomVariable
            Second partition or random variable.
        atol : float
            Input argument.
        
        Returns
        -------
        bool
            Result produced by the operation.
        
        Raises
        ------
        ValueError
            Raised when an input or mathematical precondition is violated.
        
        Examples
        --------
        See the executable examples and API reference for `are_independent`."""
        if first.space is not self or second.space is not self:
            raise ValueError("random variables must belong to this probability space")
        return self.are_partitions_independent(Partition.generated_by(first), Partition.generated_by(second), atol=atol)

    def conditional_characterization_error(self, x: RandomVariable, partition: Partition) -> float:
        """Return the maximum finite-space characterization error for ``E(X|G)``.
        
        Parameters
        ----------
        x : RandomVariable
            Random variable or numeric input.
        partition : Partition
            Conditioning partition.
        
        Returns
        -------
        float
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `conditional_characterization_error`."""
        ce = self.conditional_expectation(x, partition)
        error = 0.0
        for block in partition.blocks:
            indicator = self.random_variable({o: float(o in block) for o in self.outcomes}, name="1_A")
            error = max(error, abs((indicator * x).expectation() - (indicator * ce).expectation()))
        return float(error)

    def variance(self, x: RandomVariable) -> float:
        """Return the ordinary variance of ``x``.
        
        Parameters
        ----------
        x : RandomVariable
            Random variable or numeric input.
        
        Returns
        -------
        float
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `variance`."""
        m = x.expectation()
        return float(((x - m) * (x - m)).expectation())

    def covariance(self, x: RandomVariable, y: RandomVariable) -> float:
        """Return the ordinary covariance of ``x`` and ``y``.
        
        Parameters
        ----------
        x : RandomVariable
            Random variable or numeric input.
        y : RandomVariable
            Random variable or numeric input.
        
        Returns
        -------
        float
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `covariance`."""
        return float((x * y).expectation() - x.expectation() * y.expectation())

    def total_variance(self, x: RandomVariable, partition: Partition) -> float:
        """Return the total-variance decomposition.
        
        Parameters
        ----------
        x : RandomVariable
            Random variable or numeric input.
        partition : Partition
            Conditioning partition.
        
        Returns
        -------
        float
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `total_variance`."""
        ex = self.conditional_expectation(x, partition)
        return float(self.variance(ex) + self.conditional_variance(x, partition).expectation())

    def total_covariance(self, x: RandomVariable, y: RandomVariable, partition: Partition) -> float:
        """Return the total-covariance decomposition.
        
        Parameters
        ----------
        x : RandomVariable
            Random variable or numeric input.
        y : RandomVariable
            Random variable or numeric input.
        partition : Partition
            Conditioning partition.
        
        Returns
        -------
        float
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `total_covariance`."""
        ex = self.conditional_expectation(x, partition)
        ey = self.conditional_expectation(y, partition)
        return float(self.conditional_covariance(x, y, partition).expectation() + self.covariance(ex, ey))

    def l2_projection(self, x: RandomVariable, partition: Partition) -> RandomVariable:
        """Return the ``L2`` projection interpretation of ``E(X|G)``.
        
        Parameters
        ----------
        x : RandomVariable
            Random variable or numeric input.
        partition : Partition
            Conditioning partition.
        
        Returns
        -------
        RandomVariable
            Result produced by the operation.
        
        Raises
        ------
        ValueError
            Raised when an input or mathematical precondition is violated.
        
        Examples
        --------
        See the executable examples and API reference for `l2_projection`."""
        if self.variance(x) < -1e-12:
            raise ValueError("invalid variance")
        return self.conditional_expectation(x, partition)

    def check_measurable(self, x: RandomVariable, partition: Partition, *, atol: float = 1e-12) -> bool:
        """Check whether ``x`` is constant on every block of the partition.
        
        Parameters
        ----------
        x : RandomVariable
            Random variable or numeric input.
        partition : Partition
            Conditioning partition.
        atol : float
            Input argument.
        
        Returns
        -------
        bool
            Result produced by the operation.
        
        Examples
        --------
        See the executable examples and API reference for `check_measurable`."""
        return all(np.ptp([x.values[o] for o in block]) <= atol for block in partition.blocks)


__all__ = ["FiniteProbabilitySpace"]
