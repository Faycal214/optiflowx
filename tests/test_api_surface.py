"""Contract tests for the canonical StochX public API."""

import numpy as np

from stochx.stochastic import (
    BirthDeathProcess,
    ContinuousTimeMarkovChain,
    Filtration,
    FiniteProbabilitySpace,
    MarkovChain,
    Martingale,
    Partition,
    PoissonProcess,
    RandomVariable,
    StoppedProcess,
    StoppingTime,
)


def test_canonical_markov_api():
    chain = MarkovChain([[0.8, 0.2], [0.3, 0.7]])
    assert chain.transition_matrix_at(2).shape == (2, 2)
    assert chain.first_passage_probability(0, 1, 1) == 0.2
    assert isinstance(chain.mean_hitting_time(0, 1), float)


def test_canonical_poisson_api():
    process = PoissonProcess(2.0)
    assert process.lambda_ == 2.0
    assert process.count(1.0, rng=np.random.default_rng(1)) >= 0


def test_canonical_cmtc_api():
    ctmc = ContinuousTimeMarkovChain([[-1.0, 1.0], [2.0, -2.0]])
    assert np.allclose(ctmc.generator, ctmc.generator_matrix)
    assert ctmc.transition_matrix_at(0.0).shape == (2, 2)
    assert ctmc.forward_equation(0.0).shape == (2, 2)
    assert ctmc.backward_equation(0.0).shape == (2, 2)
    assert isinstance(ctmc.jump_chain(), MarkovChain)


def test_canonical_birth_death_api():
    birth_death = BirthDeathProcess.finite([1.0, 1.0], [0.0, 1.0])
    assert birth_death.generator.shape == (2, 2)
    assert isinstance(birth_death.jump_chain(), MarkovChain)
    assert isinstance(birth_death.to_ctmc(), ContinuousTimeMarkovChain)


def test_canonical_probability_space_object_graph():
    space = FiniteProbabilitySpace([0, 1, 2], [1 / 3] * 3)
    x = space.random_variable([0.0, 1.0, 2.0], name="X")
    y = space.random_variable([0.0, 0.0, 1.0], name="Y")
    partition = Partition.generated_by(y)

    assert isinstance(x, RandomVariable)
    assert isinstance(x + y, RandomVariable)
    assert isinstance(space.conditional_expectation(x, partition), RandomVariable)
    assert space.n_outcomes == 3
    assert partition.n_blocks == 2


def test_canonical_martingale_object_graph():
    space = FiniteProbabilitySpace([0, 1], [0.5, 0.5])
    x0 = space.random_variable([0.0, 0.0])
    x1 = space.random_variable([-1.0, 1.0])
    filtration = Filtration.natural([x0, x1])
    martingale = Martingale([x0, x1], filtration)

    assert filtration.n_steps == 2
    assert filtration.at(0) is filtration[0]
    assert martingale.n_steps == 2
    assert martingale.value_at(0) is x0

    stopping_time = StoppingTime.from_values(
        space,
        {0: 1, 1: 1},
        filtration,
    )
    stopped = martingale.stopped(stopping_time)
    assert isinstance(stopped, StoppedProcess)
    assert isinstance(stopped.values(1), RandomVariable)
