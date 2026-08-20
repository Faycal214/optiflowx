from importlib.metadata import metadata, version
from pathlib import Path

import stochx


ROOT = Path(__file__).resolve().parents[1]


def test_release_candidate_version_is_single_sourced() -> None:
    assert stochx.__version__ == "0.3.0rc1"
    assert version("stochx") == "0.3.0rc1"
    assert metadata("stochx")["Version"] == "0.3.0rc1"


def test_release_candidate_contract_is_present() -> None:
    path = ROOT / "docs" / "stage12" / "12.6_release_candidate.md"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "0.3.0rc1" in text
    assert "SHA-256" in text
    assert "Python 3.10" in text
    assert "Python 3.11" in text
    assert "Python 3.12" in text
