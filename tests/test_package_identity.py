from pathlib import Path


def test_project_identity_is_stochx() -> None:
    """Ensure the repository contains no legacy OptiFlowX identity."""
    root = Path(__file__).resolve().parents[1]

    assert (root / "stochx").is_dir()
    assert not (root / "optiflowx").exists()

    legacy_paths: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts or "site" in path.parts or "dist" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "optiflowx" in text.lower() or "OptiFlowX" in text:
            legacy_paths.append(str(path.relative_to(root)))

    assert not legacy_paths, (
        "Legacy OptiFlowX identity remains in the repository: "
        + ", ".join(sorted(legacy_paths))
    )
