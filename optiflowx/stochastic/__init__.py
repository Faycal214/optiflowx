"""Public stochastic-process API for the MSPRO Processus Aléatoires curriculum.

The package exposes domain objects directly from :mod:`optiflowx.stochastic`.
Canonical module names mirror the mathematical object names used by users;
legacy implementation modules remain internal compatibility layers during the
API migration.
"""

from .birth_death_process import BirthDeathProcess
from .continuous_time_markov_chain import CTMCPath, ContinuousTimeMarkovChain
from .filtration import Filtration
from .martingale import Martingale
from .partition import Partition
from .poisson_process import NonHomogeneousPoissonProcess, PoissonProcess
from .probability_space import FiniteProbabilitySpace
from .random_variable import RandomVariable
from .markov_chain import MarkovChain
from .stopped_process import StoppedProcess
from .stopping_time import StoppingTime
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
