"""Chapter 3: birth-death process example."""

from stochx.stochastic import BirthDeathProcess


process = BirthDeathProcess.linear(
    birth_rate=0.2,
    death_rate=0.1,
    immigration=1.0,
    max_state=6,
)

print("generator Q:\n", process.generator_matrix())
print("embedded jump-chain matrix:\n", process.jump_chain_matrix())
print("Kolmogorov derivative at p0:", process.kolmogorov_derivative([1, 0, 0, 0, 0, 0, 0]))

pure = BirthDeathProcess.pure_immigration(2.0)
print("P(X_3=4) in pure immigration:", pure.pure_immigration_probability(4, 3.0, rate=2.0))
