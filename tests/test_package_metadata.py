from importlib.metadata import metadata, version

import stochx


def test_version_is_single_sourced() -> None:
    assert stochx.__version__ == version("stochx")


def test_distribution_metadata_is_release_ready() -> None:
    info = metadata("stochx")
    assert info["Name"] == "stochx"
    assert info["Version"] == stochx.__version__
    assert "time-series" in info["Summary"].lower()
    assert "eviews" in info["Summary"].lower()
