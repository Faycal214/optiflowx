from importlib.metadata import metadata, version
from pathlib import Path

import stochx


ROOT = Path(__file__).resolve().parents[1]


def test_stable_release_version_is_single_sourced():
    assert stochx.__version__ == "0.3.0"
    assert version("stochx") == "0.3.0"
    assert metadata("stochx")["Version"] == "0.3.0"


def test_stable_release_decision_contract_is_present():
    path = ROOT / "docs" / "stage12" / "12.7_stable_release_decision.md"
    text = path.read_text(encoding="utf-8")
    assert "0.3.0" in text
    assert "Stage 8–11" in text
    assert "Python 3.10" in text
    assert "Python 3.11" in text
    assert "Python 3.12" in text
    assert "v0.3.0" in text
