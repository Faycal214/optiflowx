from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _release_workflow() -> str:
    return (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")


def test_release_workflow_has_tag_and_manual_triggers() -> None:
    workflow = _release_workflow()
    assert 'tags:' in workflow
    assert '"v*.*.*"' in workflow
    assert "workflow_dispatch:" in workflow


def test_release_workflow_gates_publish_on_verification() -> None:
    workflow = _release_workflow()
    assert "needs: verify" in workflow
    assert "if: startsWith(github.ref, 'refs/tags/v')" in workflow
    assert "Verify tag points to main" in workflow
    assert "git merge-base --is-ancestor \"$GITHUB_SHA\" origin/main" in workflow
    assert "Verify tag matches package version" in workflow
    assert 'TAG_VERSION="${GITHUB_REF_NAME#v}"' in workflow


def test_release_workflow_runs_complete_release_validation() -> None:
    workflow = _release_workflow()
    required = (
        "pytest -q tests --disable-warnings",
        "mkdocs build --strict",
        "python -m build",
        "python -m twine check dist/*",
        "python -m venv /tmp/stochx-release-venv",
        "import stochx, stochx.timeseries",
    )
    for item in required:
        assert item in workflow, f"Missing release gate: {item}"


def test_stage12_4_documentation_exists() -> None:
    assert (ROOT / "docs" / "stage12" / "12.4_ci_release_workflow_hardening.md").exists()
