"""API coverage gallery for the public OptiFlowX stochastic objects."""

from __future__ import annotations

import numpy as np

from optiflowx.stochastic import (
    BirthDeathProcess,
    CTMCPath,
    ContinuousTimeMarkovChain,
    FiniteProbabilitySpace,
    Filtration,
    MarkovChain,
    Martingale,
    NonHomogeneousPoissonProcess,
    Partition,
    PoissonProcess,
    RandomVariable,
    StoppedProcess,
    StoppingTime,
    empirical_state_frequencies,
)


def markov_chain_examples() -> None:
    chain = MarkovChain([[0.7, 0.3], [0.4, 0.6]], states=["A", "B"])
    print(chain.states, chain.n_states, chain.transition_matrix)
    print(chain.n_step_transition(2), chain.transition_matrix_at(2))
    print(chain.state_distribution([1.0, 0.0], 2))
    print(chain.chapman_kolmogorov(1, 2))
    print(chain.transition_graph())
    print(chain.accessible("A", "B"), chain.communicate("A", "B"))
    print(chain.communicating_classes(), chain.closed_classes())
    print(chain.is_irreducible(), chain.is_absorbing_state("A"), chain.classify_states())
    print(chain.first_visit_probability("A", "B", 2))
    print(chain.first_passage_probability("A", "B", 2))
    print(chain.visit_probability("A", "B", 3))
    print(chain.hitting_probability("A", "B"))
    print(chain.expected_hitting_time("A", "B"), chain.mean_hitting_time("A", "B"))
    print(chain.first_return_probability("A", 2), chain.return_probability("A"))
    print(chain.mean_return_time("A"))
    print(chain.period("A"), chain.is_aperiodic(), chain.is_ergodic())
    print(chain.stationary_distribution(), chain.stationary_distributions())
    print(chain.limiting_distribution(), chain.jump_chain() is chain)
    path = chain.simulate(5, initial_state="A", rng=np.random.default_rng(0))
    print(path)
    print(empirical_state_frequencies(path, chain.states))

    absorbing = MarkovChain(
        [[1.0, 0.0, 0.0], [0.2, 0.5, 0.3], [0.0, 0.0, 1.0]],
        states=["A", "B", "C"],
    )
    print(absorbing.absorption_probability("B", ["A"]))


def poisson_examples() -> None:
    process = PoissonProcess(2.0)
    print(process.rate, process.lambda_)
    print(process.count_probability(3, 2.0))
    print(process.increment_probability(2, 1.0, 3.0))
    print(process.interarrival_samples(4, rng=np.random.default_rng(1)))
    print(process.arrival_times(4, rng=np.random.default_rng(1)))
    print(process.simulate(3.0, rng=np.random.default_rng(1)))
    print(process.count_sample(2.0, rng=np.random.default_rng(1)), process.count(2.0, rng=np.random.default_rng(1)))
    print(process.conditional_first_arrival_cdf(1.0, 2.0))
    print(process.conditional_arrival_times(4, 2.0, rng=np.random.default_rng(1)))
    print(process.superpose(PoissonProcess(1.0)).rate)
    kept, rejected = process.split(0.4)
    print(kept.rate, rejected.rate)

    nhpp = NonHomogeneousPoissonProcess(
        intensity=lambda t: 1.0 + t,
        mean_function=lambda t: t + 0.5 * t**2,
    )
    print(nhpp.intensity_function(2.0), nhpp.mean(2.0))
    print(nhpp.count_probability(2, 2.0), nhpp.increment_probability(1, 0.5, 2.0))


def ctmc_examples() -> None:
    chain = ContinuousTimeMarkovChain([[-2.0, 2.0], [1.0, -1.0]], states=["A", "B"])
    print(chain.states, chain.n_states, chain.generator_matrix, chain.generator)
    print(chain.infinitesimal_transition_matrix(1e-3))
    print(chain.transition_matrix(2.0))
    print(chain.transition_matrix_at(2.0, method="uniformization"))
    print(chain.transition_matrix_uniformized(2.0))
    print(chain.transition_probability("A", "B", 2.0))
    print(chain.state_distribution([1.0, 0.0], 2.0))
    print(chain.chapman_kolmogorov(1.0, 2.0))
    print(chain.forward_derivative(1.0), chain.forward_equation(1.0))
    print(chain.backward_derivative(1.0), chain.backward_equation(1.0))
    print(chain.stationary_distribution(), chain.communicating_classes())
    print(chain.stationary_distribution_from_jump_chain(), chain.mean_return_time("A"))
    print(chain.long_run_cost([5.0, 1.0]))
    print(chain.holding_rate("A"), chain.holding_time("A", rng=np.random.default_rng(2)))
    print(chain.jump_chain_matrix(), chain.jump_chain().states)
    path = chain.simulate(5.0, initial_state="A", rng=np.random.default_rng(2))
    print(isinstance(path, CTMCPath), path.times, path.states, path.state_at(2.0))
    print(path.occupation_time("A", 5.0), path.occupation_fraction("A", 5.0))


def birth_death_examples() -> None:
    process = BirthDeathProcess.linear(
        birth_rate=0.2,
        death_rate=0.1,
        immigration=1.0,
        max_state=6,
    )
    print(process.max_state, process.birth_rate(2), process.death_rate(2))
    print(process.generator, process.generator_matrix(), process.jump_chain_matrix())
    print(process.to_ctmc().states, process.jump_chain().states)
    print(process.kolmogorov_derivative([1.0] + [0.0] * 6))
    print(process.stationary_weights(6), process.stationary_weights_at(6))
    print(process.stationary_distribution())

    finite = BirthDeathProcess.finite([0.5, 0.7, 0.9], [0.0, 0.2, 0.4])
    print(finite.generator_matrix())
    print(BirthDeathProcess.pure_immigration(1.0).generator_matrix())
    print(BirthDeathProcess.pure_immigration_probability(3, 2.0, rate=1.0))
    print(BirthDeathProcess.pure_birth_probability(3, 2.0, rate=0.5))
    print(BirthDeathProcess.pure_death_probability(3, 2.0, initial_population=5, rate=0.3))
    print(BirthDeathProcess.pure_birth(0.4).pure_birth_reciprocal_rate_sum(5))
    print(BirthDeathProcess.pure_death(0.3).generator_matrix())


def probability_and_partition_examples() -> tuple[FiniteProbabilitySpace, Partition]:
    space = FiniteProbabilitySpace(["H", "T"], [0.6, 0.4])
    X = space.random_variable({"H": 1.0, "T": 0.0}, name="X")
    Y = space.random_variable({"H": 2.0, "T": 3.0}, name="Y")
    G = Partition.from_blocks([{"H"}, {"T"}], space)

    print(space.outcomes, space.probabilities, space.n_outcomes)
    print(space.probability({"H"}), space.probability_of({"H"}))
    print(isinstance(X, RandomVariable), X.array(), X.expectation(), X.expected_value(), X.support)
    print(X.transform(lambda x: x + 1.0, name="Z").array())
    print(X.apply(lambda x: 2.0 * x).array())
    print((X + Y).array(), (X - Y).array(), (X * Y).array(), (-X).array())
    print(G.blocks, G.n_blocks, G.contains("H"), G.refines(G))
    print(Partition.generated_by(Y).blocks)
    print(space.conditional_probability_given_event({"H"}, {"H", "T"}))
    print(space.conditional_expectation_given_event(X, {"H", "T"}).array())
    print(space.conditional_expectation(X, G).array())
    print(space.conditional_expectation_given(X, Y).array())
    print(space.conditional_probability({"H"}, G).array())
    print(space.are_partitions_independent(G, G), space.are_independent(X, Y))
    print(space.conditional_characterization_error(X, G))
    print(space.total_expectation(X, G))
    print(space.tower(X, G, G).array())
    print(space.pull_out(Y, X, G).array())
    print(space.conditional_variance(X, G).array())
    print(space.conditional_covariance(X, Y, G).array())
    print(space.total_variance(X, G), space.total_covariance(X, Y, G))
    print(space.l2_projection(X, G).array())
    return space, G


def martingale_examples(space: FiniteProbabilitySpace) -> None:
    x0 = space.random_variable([0.0, 0.0], name="X0")
    x1 = space.random_variable([1.0, -1.0], name="X1")
    process = [x0, x1]
    filtration = Filtration.natural(process)
    print(filtration.partitions, filtration.n_steps)
    print(filtration.at(0), filtration.is_adapted(process))

    mart = Martingale(process, filtration)
    print(mart.process, mart.filtration, mart.n_steps)
    print(mart.value_at(0).array())
    print(mart.conditional_next(0).array())
    print(mart.martingale_residual(0).array())
    print(mart.is_martingale(), mart.is_submartingale(), mart.is_supermartingale())
    print(mart.conditional_future(0, 1).array(), mart.expectations())
    print(mart.transform(abs).is_submartingale())
    print(Martingale.doob(x1, filtration).is_martingale())

    stopping = StoppingTime.from_values(space, {"H": 1, "T": 1}, filtration)
    print(stopping.space, stopping.values, stopping.filtration)
    print(stopping.minimum(stopping).values)
    print(stopping.maximum(stopping).values)
    print(stopping.add(stopping).values)
    stopped = mart.stopped(stopping)
    print(isinstance(stopped, StoppedProcess), stopped.process, stopped.stopping_time, stopped.n_steps)
    print(stopped.values(1), stopped.sequence(), stopped.terminal_value())


if __name__ == "__main__":
    markov_chain_examples()
    poisson_examples()
    ctmc_examples()
    birth_death_examples()
    space, _ = probability_and_partition_examples()
    martingale_examples(space)
