from pathlib import Path

import stochx


ROOT = Path(__file__).resolve().parents[1]


def test_stage12_release_surface_documents_current_scope() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")

    assert "Box–Jenkins" in readme
    assert "state-space" in readme
    assert "Kalman" in readme
    assert "09_state_space_kalman.py" in readme
    assert "Stage 11" in changelog
    assert "Box–Jenkins" in changelog
    assert "Kalman" in changelog
    assert "time-series" in citation.lower()
    assert "state-space" in citation.lower()


def test_stage12_release_examples_exist_and_are_current() -> None:
    required = (
        "01_discrete_markov_chain.py",
        "02_poisson_process.py",
        "03_continuous_markov_chain.py",
        "04_birth_death_process.py",
        "05_conditional_expectation.py",
        "06_martingale.py",
        "07_api_operations.py",
        "09_state_space_kalman.py",
        "10_state_space_workflow.py",
    )
    for filename in required:
        assert (ROOT / "examples" / filename).exists(), f"Missing example: {filename}"


def test_stage12_release_documentation_files_exist() -> None:
    assert (ROOT / "docs" / "stage12" / "12.1_release_readiness.md").exists()
    assert (ROOT / "docs" / "stage12" / "12.2_distribution_hardening.md").exists()
    assert (ROOT / "docs" / "stage12" / "12.3_release_surface_closure.md").exists()
    assert (ROOT / "docs" / "stage10" / "10.2_state_space_kalman.md").exists()
    assert (ROOT / "docs" / "stage11" / "11.10_stage_freeze.md").exists()
    assert (ROOT / "docs" / "time_series.md").exists()
    assert (ROOT / "docs" / "examples.md").exists()


def test_stage12_version_remains_single_sourced_before_release_candidate() -> None:
    init_text = (ROOT / "stochx" / "__init__.py").read_text(encoding="utf-8")
    assert f'__version__ = "{stochx.__version__}"' in init_text
    assert stochx.__version__ == "0.2.0"
