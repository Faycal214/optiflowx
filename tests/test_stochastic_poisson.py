import numpy as np
import pytest

from optiflowx.stochastic import NonHomogeneousPoissonProcess, PoissonProcess


def test_poisson_count_probability():
    process = PoissonProcess(2.0)
    assert process.count_probability(0, 0.0) == pytest.approx(1.0)
    assert process.count_probability(4, 3.0) == pytest.approx(np.exp(-6.0) * 6.0**4 / 24.0)
    assert process.increment_probability(3, 1.0, 3.0) == pytest.approx(
        np.exp(-4.0) * 4.0**3 / 6.0
    )


def test_interarrival_times_are_generated():
    process = PoissonProcess(2.0)
    sample = process.interarrival_samples(1000, rng=np.random.default_rng(123))
    assert sample.shape == (1000,)
    assert np.all(sample > 0)


def test_arrival_times_are_increasing():
    process = PoissonProcess(2.0)
    times = process.arrival_times(50, rng=np.random.default_rng(123))
    assert np.all(np.diff(times) > 0)


def test_simulation_and_direct_count():
    process = PoissonProcess(2.0)
    times = process.simulate(10.0, rng=np.random.default_rng(123))
    assert np.all(times <= 10.0)
    assert np.all(np.diff(times) > 0)

    count = process.count_sample(10.0, rng=np.random.default_rng(123))
    assert isinstance(count, int)


def test_conditional_first_arrival_is_uniform():
    process = PoissonProcess(2.0)
    assert process.conditional_first_arrival_cdf(-1.0, 3.0) == 0.0
    assert process.conditional_first_arrival_cdf(1.5, 3.0) == pytest.approx(0.5)
    assert process.conditional_first_arrival_cdf(4.0, 3.0) == 1.0


def test_conditional_k_arrival_times_are_ordered_uniform_samples():
    process = PoissonProcess(2.0)
    times = process.conditional_arrival_times(10, 5.0, rng=np.random.default_rng(123))
    assert times.shape == (10,)
    assert np.all(times >= 0.0)
    assert np.all(times <= 5.0)
    assert np.all(np.diff(times) >= 0.0)


def test_superposition_has_sum_rate():
    combined = PoissonProcess(2.0).superpose(PoissonProcess(3.0))
    assert combined.rate == pytest.approx(5.0)


def test_thinning_rates():
    first, second = PoissonProcess(10.0).split(0.3)
    assert first.rate == pytest.approx(3.0)
    assert second.rate == pytest.approx(7.0)


def test_nonhomogeneous_poisson_process_uses_mean_function():
    process = NonHomogeneousPoissonProcess(
        intensity=lambda t: 2.0 * t,
        mean_function=lambda t: t**2,
    )
    assert process.mean(3.0) == pytest.approx(9.0)
    assert process.count_probability(2, 3.0) == pytest.approx(np.exp(-9.0) * 9.0**2 / 2.0)
    assert process.increment_probability(1, 1.0, 3.0) == pytest.approx(np.exp(-8.0) * 8.0)


def test_nonhomogeneous_mean_can_be_integrated_numerically():
    process = NonHomogeneousPoissonProcess(lambda t: 2.0 * t)
    assert process.mean(3.0) == pytest.approx(9.0, abs=1e-3)


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        PoissonProcess(0.0)
    with pytest.raises(ValueError):
        PoissonProcess(2.0).count_probability(-1, 1.0)
    with pytest.raises(ValueError):
        PoissonProcess(2.0).increment_probability(1, 3.0, 1.0)
    with pytest.raises(ValueError):
        PoissonProcess(2.0).conditional_first_arrival_cdf(1.0, 0.0)
