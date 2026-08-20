from pathlib import Path


REQUIRED_DOCUMENTATION = (
    "index.md",
    "about.md",
    "course_material.md",
    "course_chapitre1.md",
    "course_chapitre2.md",
    "course_chapitre3.md",
    "course_chapitre4.md",
    "course_chapitre5.md",
    "architecture.md",
    "api.md",
    "api/index.md",
    "api/time-series.md",
    "api/stochastic-processes.md",
    "course-map.md",
    "examples.md",
    "time_series.md",
    "time-series/index.md",
    "time-series/eviews-workflow.md",
    "time-series/data-series.md",
    "time-series/transforms.md",
    "time-series/stationarity.md",
    "time-series/correlation.md",
    "time-series/models.md",
    "time-series/box-jenkins.md",
    "time-series/diagnostics.md",
    "time-series/forecasting.md",
    "time-series/state-space.md",
    "time-series/reports.md",
    "stochastic/index.md",
    "stochastic/markov-chains.md",
    "stochastic/poisson-processes.md",
    "stochastic/ctmc-birth-death.md",
    "stochastic/probability-objects.md",
    "stochastic/martingales.md",
    "stochastic/simulation.md",
)

REQUIRED_EXAMPLES = (
    "01_discrete_markov_chain.py",
    "02_poisson_process.py",
    "03_continuous_markov_chain.py",
    "04_birth_death_process.py",
    "05_conditional_expectation.py",
    "06_martingale.py",
    "07_api_operations.py",
    "08_eviews_time_series_workflow.py",
    "09_state_space_kalman.py",
    "10_state_space_workflow.py",
)


def test_required_documentation_pages_exist() -> None:
    missing = [path for path in REQUIRED_DOCUMENTATION if not Path("docs", path).exists()]
    assert not missing, "Missing documentation pages: " + ", ".join(missing)


def test_required_runnable_examples_exist() -> None:
    missing = [path for path in REQUIRED_EXAMPLES if not Path("examples", path).exists()]
    assert not missing, "Missing runnable examples: " + ", ".join(missing)
