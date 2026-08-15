"""Complete Chapter 2 Poisson-process API demonstration."""

import numpy as np

from stochx.stochastic import NonHomogeneousPoissonProcess, PoissonProcess

rng = np.random.default_rng(7)
pp = PoissonProcess(2.0)

print("count:", pp.count_probability(4, 2.0))
print("increment:", pp.increment_probability(2, 1.0, 3.0))
print("sample count:", pp.count_sample(2.0, rng=rng))
print("inter-arrivals:", pp.interarrival_samples(5, rng=rng))
print("arrival times:", pp.arrival_times(5, rng=rng))
print("simulated events:", pp.simulate(5.0, rng=rng))
print("conditional first-arrival CDF:", pp.conditional_first_arrival_cdf(1.0, 2.0))
print("conditional arrivals:", pp.conditional_arrival_times(4, 10.0, rng=rng))

combined = pp.superpose(PoissonProcess(3.0))
print("superposed rate:", combined.rate)
first, second = pp.split(0.4)
print("split rates:", first.rate, second.rate)

nhpp = NonHomogeneousPoissonProcess(
    intensity=lambda t: 2.0 * t,
    mean_function=lambda t: t**2,
)
print("NHPP mean:", nhpp.mean(3.0))
print("NHPP count probability:", nhpp.count_probability(2, 3.0))
print("NHPP increment probability:", nhpp.increment_probability(1, 1.0, 3.0))
