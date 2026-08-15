"""Complete Chapter 3 CMTC API demonstration."""

import numpy as np

from optiflowx.stochastic import ContinuousTimeMarkovChain

Q = np.array([[-2.0, 2.0], [1.0, -1.0]])
chain = ContinuousTimeMarkovChain(Q, states=["A", "B"])

print("Q =\n", chain.generator_matrix)
print("I+hQ =\n", chain.infinitesimal_transition_matrix(1e-3))
print("P(2) =\n", chain.transition_matrix(2.0))
print("p_AB(2) =", chain.transition_probability("A", "B", 2.0))
print("law at t=2 =", chain.state_distribution([1.0, 0.0], 2.0))
print("P(1)P(2) =\n", chain.chapman_kolmogorov(1.0, 2.0))
print("forward derivative =\n", chain.forward_derivative(2.0))
print("backward derivative =\n", chain.backward_derivative(2.0))
print("stationary =", chain.stationary_distribution())
print("stationary from jump chain =", chain.stationary_distribution_from_jump_chain())
print("communication classes =", chain.communicating_classes())
print("mean return time A =", chain.mean_return_time("A"))
print("long-run cost =", chain.long_run_cost([10.0, 2.0]))
print("holding rate A =", chain.holding_rate("A"))
print("holding time A =", chain.holding_time("A", rng=np.random.default_rng(7)))
print("jump chain =\n", chain.jump_chain_matrix())

path = chain.simulate(10.0, initial_state="A", rng=np.random.default_rng(7))
print("jump times =", path.times)
print("visited states =", path.states)
print("state at 4 =", path.state_at(4.0))
print("occupation time A =", path.occupation_time("A", 10.0))
print("occupation fraction A =", path.occupation_fraction("A", 10.0))
