"""Chapter 3: continuous-time Markov-chain example."""

import numpy as np

from optiflowx.stochastic import (
    ContinuousTimeMarkovChain,
    ctmc_mean_return_time,
    ctmc_stationary_from_jump_chain,
    occupation_fraction,
)


chain = ContinuousTimeMarkovChain(
    [[-2.0, 2.0], [1.0, -1.0]],
    states=["A", "B"],
)

print("P(0.5):\n", chain.transition_matrix(0.5))
print("stationary distribution:", chain.stationary_distribution())
print("stationary distribution from jump chain:", ctmc_stationary_from_jump_chain(chain))
print("mean return time to A:", ctmc_mean_return_time(chain, "A"))

path = chain.simulate(10.0, initial_state="A", rng=np.random.default_rng(42))
print("occupation fraction of A over [0,10]:", occupation_fraction(path, "A", 10.0))
