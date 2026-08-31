from importlib.metadata import metadata, version
from pathlib import Path
import re

import stochx
import stochx.timeseries as timeseries


def test_release_version_is_single_sourced():
    assert stochx.__version__ == version("stochx")
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:rc\d+)?", stochx.__version__)


def test_distribution_identity_is_consistent():
    info = metadata("stochx")
    assert info["Name"] == "stochx"
    assert info["Version"] == stochx.__version__
    assert info["License"]
    assert "time series" in info["Summary"].lower()
    assert info["Requires-Python"] == ">=3.10"


def test_top_level_public_namespace_is_time_series_only():
    assert stochx.__all__ == ["timeseries"]
    assert not hasattr(stochx, "stochastic")
    assert timeseries is stochx.timeseries


def test_frozen_public_api_remains_importable():
    required = (
        "TimeSeries",
        "Workfile",
        "Equation",
        "fit_arima",
        "autoarma",
        "johansen",
        "vecm",
    )
    for name in required:
        assert hasattr(timeseries, name), f"Missing frozen public API: {name}"


def test_release_documents_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "docs" / "stage12" / "12.1_release_readiness.md").exists()
    assert (root / "CHANGELOG.md").exists()
    assert (root / "CITATION.cff").exists()
    assert (root / "pyproject.toml").exists()


def test_release_metadata_contains_release_links():
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "stochx"' in pyproject
    assert 'requires-python = ">=3.10"' in pyproject
    assert 'Homepage = "https://github.com/Faycal214/stochx"' in pyproject
    assert 'Documentation = "https://faycal214.github.io/stochx/"' in pyproject
    assert 'Repository = "https://github.com/Faycal214/stochx"' in pyproject
