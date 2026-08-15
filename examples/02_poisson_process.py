"""Chapter 2: homogeneous Poisson-process example."""

import numpy as np

from stochx.stochastic import PoissonProcess


process = PoissonProcess(rate=2.0)

print("P(N(3)=4):", process.count_probability(4, 3.0))
print("P(N(3)-N(1)=3):", process.increment_probability(3, 1.0, 3.0))

interarrivals = process.interarrival_samples(5, rng=np.random.default_rng(42))
print("five inter-arrival times:", interarrivals)
print("five arrival times:", np.cumsum(interarrivals))

conditional_times = process.conditional_arrival_times(
    4,
    3.0,
    rng=np.random.default_rng(42),
)
print("arrival times conditional on N(3)=4:", conditional_times)
