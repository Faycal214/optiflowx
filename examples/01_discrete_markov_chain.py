"""Chapter 1: discrete-time Markov chain example.

The matrix P is the one-step transition matrix.  P**n gives n-step
transition probabilities and a stationary distribution pi satisfies pi P=pi.
"""

import numpy as np

from optiflowx.stochastic import MarkovChain, empirical_state_frequencies, mean_return_time


chain = MarkovChain(
    [[0.7, 0.3], [0.4, 0.6]],
    states=["A", "B"],
)

print("P^5:\n", chain.n_step_transition(5))
print("stationary distribution:", chain.stationary_distribution())
print("mean return time to A:", mean_return_time(chain, "A"))

path = chain.simulate(
    10_000,
    initial_state="A",
    rng=np.random.default_rng(42),
)
print("empirical frequencies:", empirical_state_frequencies(path, chain.states))
