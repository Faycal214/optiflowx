from importlib.metadata import metadata, version
from pathlib import Path
import re

import stochx
import stochx.stochastic as stochastic
import stochx.timeseries as timeseries


def test_stage12_package_version_is_single_sourced():
    assert stochx.__version__ == version("stochx")
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:rc\d+)?", stochx.__version__)


def test_stage12_distribution_identity_is_consistent():
    info = metadata("stochx")
    assert info["Name"] == "stochx"
    assert info["Version"] == stochx.__version__
    assert info["License"]
    assert "stochastic-process" in info["Summary"]
    assert info["Requires-Python"] == ">=3.10"


def test_stage12_top_level_public_namespaces_remain_available():
    assert stochastic is stochx.stochastic
    assert timeseries is stochx.timeseries
    assert stochx.__all__ == ["stochastic", "timeseries"]


def test_stage12_frozen_state_space_public_api_remains_importable():
    required = (
        "LinearStateSpace",
        "KalmanFilterResult",
        "kalman_filter",
        "local_level_filter",
        "KalmanSmootherResult",
        "kalman_smoother",
        "KalmanForecastResult",
        "kalman_forecast",
        "LocalLevelEstimateResult",
        "estimate_local_level",
        "KalmanInnovationDiagnosticsResult",
        "kalman_innovation_diagnostics",
        "StateSpaceAdequacyResult",
        "state_space_adequacy",
        "StateSpaceWorkflowResult",
        "run_local_level_workflow",
    )
    for name in required:
        assert hasattr(timeseries, name), f"Missing frozen public API: {name}"


def test_stage12_release_documents_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "docs" / "stage12" / "12.1_release_readiness.md").exists()
    assert (root / "CHANGELOG.md").exists()
    assert (root / "CITATION.cff").exists()
    assert (root / "pyproject.toml").exists()


def test_stage12_release_metadata_contains_release_links():
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "stochx"' in pyproject
    assert 'requires-python = ">=3.10"' in pyproject
    assert 'Homepage = "https://github.com/Faycal214/stochx"' in pyproject
    assert 'Documentation = "https://faycal214.github.io/stochx/"' in pyproject
    assert 'Repository = "https://github.com/Faycal214/stochx"' in pyproject
