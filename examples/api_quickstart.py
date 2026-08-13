"""Minimal public API syntax example.

Run from the repository root after installing OptiFlowX with `pip install -e .`.
"""

import numpy as np

from optiflowx.stochastic import MarkovChain, PoissonProcess


P = np.array([[0.8, 0.2], [0.1, 0.9]])
chain = MarkovChain(P, states=["A", "B"])
print(chain.stationary_distribution())

poisson = PoissonProcess(rate=2.0)
print(poisson.count_probability(3, 1.0))
