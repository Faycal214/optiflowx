import pytest

from stochx.stochastic import BirthDeathProcess


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BirthDeathProcess.pure_immigration(1.0),
        lambda: BirthDeathProcess.pure_birth(0.5),
        lambda: BirthDeathProcess.pure_death(0.3),
    ],
)
def test_pure_factories_remain_unbounded(factory) -> None:
    process = factory()

    assert process.max_state is None
    with pytest.raises(ValueError, match="max_state is required"):
        process.generator_matrix()


def test_finite_generator_requires_and_uses_max_state() -> None:
    process = BirthDeathProcess(
        lambda _k: 1.0,
        lambda _k: 0.5,
        max_state=3,
    )

    generator = process.generator_matrix()

    assert process.max_state == 3
    assert generator.shape == (4, 4)
    assert generator[0, 1] == pytest.approx(1.0)
    assert generator[1, 0] == pytest.approx(0.5)
    assert generator[3, 3] == pytest.approx(-0.5)
