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
    "api.md",
    "api/index.md",
    "api/architecture.md",
    "examples.md",
)

REQUIRED_EXAMPLES = (
    "01_discrete_markov_chain.py",
    "02_poisson_process.py",
    "03_continuous_markov_chain.py",
    "04_birth_death_process.py",
    "05_conditional_expectation.py",
    "06_martingale.py",
    "07_api_operations.py",
)


def test_required_documentation_pages_exist() -> None:
    missing = [path for path in REQUIRED_DOCUMENTATION if not Path("docs", path).exists()]
    assert not missing, "Missing documentation pages: " + ", ".join(missing)


def test_required_runnable_examples_exist() -> None:
    missing = [path for path in REQUIRED_EXAMPLES if not Path("examples", path).exists()]
    assert not missing, "Missing runnable examples: " + ", ".join(missing)
