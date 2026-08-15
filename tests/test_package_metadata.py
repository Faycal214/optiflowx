from importlib.metadata import metadata, version

import optiflowx


def test_version_is_single_sourced() -> None:
    assert optiflowx.__version__ == version("optiflowx")


def test_distribution_metadata_is_release_ready() -> None:
    info = metadata("optiflowx")
    assert info["Name"] == "optiflowx"
    assert info["Version"] == optiflowx.__version__
    assert "stochastic-process" in info["Summary"]
