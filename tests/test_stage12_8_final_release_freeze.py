from importlib.metadata import metadata, version
from pathlib import Path

import stochx


ROOT = Path(__file__).resolve().parents[1]


def test_final_release_freeze_contract():
    assert stochx.__version__ == "0.3.0"
    assert version("stochx") == "0.3.0"
    assert metadata("stochx")["Version"] == "0.3.0"

    path = ROOT / "docs" / "stage12" / "12.8_final_release_freeze.md"
    text = path.read_text(encoding="utf-8")
    assert "0.3.0" in text
    assert "Stage 8" in text
    assert "Stage 9" in text
    assert "Stage 10" in text
    assert "Stage 11" in text
    assert "Python 3.10" in text
    assert "Python 3.11" in text
    assert "Python 3.12" in text
    assert "v0.3.0" in text
