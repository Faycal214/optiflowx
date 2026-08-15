"""Chapter 3: continuous-time Markov-chain example."""

import numpy as np

from optiflowx.stochastic import ContinuousTimeMarkovChain

chain = ContinuousTimeMarkovChain(
    [[-2.0, 2.0], [1.0, -1.0]],
    states=["A", "B"],
)

print("P(0.5):\n", chain.transition_matrix(0.5))
print("stationary distribution:", chain.stationary_distribution())
print("stationary distribution from jump chain:", chain.stationary_distribution_from_jump_chain())
print("mean return time to A:", chain.mean_return_time("A"))

path = chain.simulate(10.0, initial_state="A", rng=np.random.default_rng(42))
print("occupation fraction of A over [0,10]:", path.occupation_fraction("A", 10.0))
