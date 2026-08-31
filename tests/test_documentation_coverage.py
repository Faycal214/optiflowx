from pathlib import Path


REQUIRED_DOCUMENTATION = (
    "index.md",
    "about.md",
    "architecture.md",
    "api.md",
    "api/index.md",
    "api/time-series.md",
    "course-material.md",
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
)


REQUIRED_EXAMPLES = (
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
