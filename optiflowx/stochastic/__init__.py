"""Stochastic-process tools for the MSPRO Processus Aléatoires curriculum."""

from .birth_death import BirthDeathProcess
from .cmtc import CTMCPath, ContinuousTimeMarkovChain
from .conditional import FiniteProbabilitySpace, Partition, RandomVariable
from .markov import MarkovChain
from .martingale import Filtration, Martingale, StoppedProcess, StoppingTime
from .poisson import NonHomogeneousPoissonProcess, PoissonProcess
from .theory import (
    conditional_characterization_error,
    conditional_expectation_given_event,
    conditional_probability_given_event,
    ctmc_communication_classes,
    ctmc_mean_return_time,
    ctmc_stationary_from_jump_chain,
    empirical_state_frequencies,
    first_return_probability,
    independent_partitions,
    independent_random_variables,
    mean_return_time,
    occupation_fraction,
    occupation_time,
    return_probability,
    stationary_distributions,
    stopped_martingale,
    transform_martingale,
)

__all__ = [
    "BirthDeathProcess",
    "CTMCPath",
    "ContinuousTimeMarkovChain",
    "FiniteProbabilitySpace",
    "Filtration",
    "MarkovChain",
    "Martingale",
    "NonHomogeneousPoissonProcess",
    "Partition",
    "PoissonProcess",
    "RandomVariable",
    "StoppedProcess",
    "StoppingTime",
    "conditional_characterization_error",
    "conditional_expectation_given_event",
    "conditional_probability_given_event",
    "ctmc_communication_classes",
    "ctmc_mean_return_time",
    "ctmc_stationary_from_jump_chain",
    "empirical_state_frequencies",
    "first_return_probability",
    "independent_partitions",
    "independent_random_variables",
    "mean_return_time",
    "occupation_fraction",
    "occupation_time",
    "return_probability",
    "stationary_distributions",
    "stopped_martingale",
    "transform_martingale",
]
