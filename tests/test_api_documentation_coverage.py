from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path

import optiflowx.stochastic as stochastic


API_ROOT = Path("docs/api")

EXPECTED_PAGES = {
    "BirthDeathProcess": "birth_death_process.md",
    "CTMCPath": "ctmc_path.md",
    "ContinuousTimeMarkovChain": "continuous_time_markov_chain.md",
    "FiniteProbabilitySpace": "finite_probability_space.md",
    "Filtration": "filtration.md",
    "MarkovChain": "markov_chain.md",
    "Martingale": "martingale.md",
    "Partition": "partition.md",
    "NonHomogeneousPoissonProcess": "non_homogeneous_poisson_process.md",
    "PoissonProcess": "poisson_process.md",
    "RandomVariable": "random_variable.md",
    "StoppedProcess": "stopped_process.md",
    "StoppingTime": "stopping_time.md",
}

EXCEPTION_PAGE = "exceptions.md"
ANALYSIS_PAGE = "analysis.md"


def _public_objects_from_stochastic_modules() -> tuple[dict[str, type], dict[str, object]]:
    classes: dict[str, type] = {}
    functions: dict[str, object] = {}

    module_names = [stochastic.__name__]
    if hasattr(stochastic, "__path__"):
        module_names.extend(
            info.name
            for info in pkgutil.iter_modules(stochastic.__path__, stochastic.__name__ + ".")
            if not info.name.rsplit(".", 1)[-1].startswith("_")
        )

    for module_name in module_names:
        module = importlib.import_module(module_name)
        for name, obj in inspect.getmembers(module):
            if name.startswith("_"):
                continue
            if getattr(obj, "__module__", None) != module.__name__:
                continue
            if inspect.isclass(obj):
                classes[name] = obj
            elif inspect.isfunction(obj):
                functions[name] = obj

    return classes, functions


def test_every_public_stochastic_class_has_an_api_page() -> None:
    classes, _ = _public_objects_from_stochastic_modules()
    missing = [
        f"{name} -> {EXPECTED_PAGES[name]}"
        for name in classes
        if name in EXPECTED_PAGES and not (API_ROOT / EXPECTED_PAGES[name]).exists()
    ]
    assert not missing, "Missing API pages: " + ", ".join(sorted(missing))


def test_api_pages_mention_the_documented_class() -> None:
    classes, _ = _public_objects_from_stochastic_modules()
    missing_mentions = []
    for name, filename in EXPECTED_PAGES.items():
        if name not in classes:
            continue
        page = API_ROOT / filename
        if not page.exists():
            continue
        text = page.read_text(encoding="utf-8")
        if f"# {name}" not in text and name not in text:
            missing_mentions.append(f"{name} -> {filename}")
    assert not missing_mentions, "API pages missing class names: " + ", ".join(sorted(missing_mentions))


def test_public_analysis_function_has_api_page() -> None:
    _, functions = _public_objects_from_stochastic_modules()
    if "empirical_state_frequencies" not in functions:
        return
    page = API_ROOT / ANALYSIS_PAGE
    assert page.exists(), f"Missing API page: {ANALYSIS_PAGE}"
    assert "empirical_state_frequencies" in page.read_text(encoding="utf-8")


def test_public_exception_classes_have_exception_reference() -> None:
    classes, _ = _public_objects_from_stochastic_modules()
    exception_names = [name for name, obj in classes.items() if issubclass(obj, Exception)]
    if not exception_names:
        return
    page = API_ROOT / EXCEPTION_PAGE
    assert page.exists(), f"Missing exception reference: {EXCEPTION_PAGE}"
    text = page.read_text(encoding="utf-8")
    missing = [name for name in exception_names if name not in text]
    assert not missing, "Exception reference missing: " + ", ".join(sorted(missing))
