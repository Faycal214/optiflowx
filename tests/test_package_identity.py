from pathlib import Path


def test_project_identity_is_stochx() -> None:
    """Ensure the repository contains no legacy project identity."""
    root = Path(__file__).resolve().parents[1]
    legacy_name = "opt" + "iflowx"

    assert (root / "stochx").is_dir()
    assert not (root / legacy_name).exists()

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
        if legacy_name in text.lower():
            legacy_paths.append(str(path.relative_to(root)))

    assert not legacy_paths, (
        "Legacy project identity remains in the repository: "
        + ", ".join(sorted(legacy_paths))
    )


def test_public_package_boundary_is_time_series_only() -> None:
    import stochx

    assert hasattr(stochx, "timeseries")
    assert not hasattr(stochx, "stochastic")
