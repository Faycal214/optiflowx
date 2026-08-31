from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path

import stochx.timeseries as timeseries


API_ROOT = Path("docs/api")


def _public_objects_from_timeseries_modules() -> tuple[dict[str, type], dict[str, object]]:
    classes: dict[str, type] = {}
    functions: dict[str, object] = {}
    module_names = [timeseries.__name__]
    module_names.extend(
        info.name
        for info in pkgutil.iter_modules(
            timeseries.__path__, timeseries.__name__ + "."
        )
        if not info.name.rsplit(".", 1)[-1].startswith("_")
    )
    for module_name in module_names:
        module = importlib.import_module(module_name)
        for name, obj in inspect.getmembers(module):
            if name.startswith("_") or getattr(obj, "__module__", None) != module.__name__:
                continue
            if inspect.isclass(obj):
                classes[name] = obj
            elif inspect.isfunction(obj):
                functions[name] = obj
    return classes, functions


def test_public_timeseries_modules_import_cleanly():
    classes, functions = _public_objects_from_timeseries_modules()
    assert classes or functions


def test_time_series_api_documentation_exists():
    page = API_ROOT / "time-series.md"
    assert page.exists(), f"Missing API page: {page}"


def test_time_series_api_page_mentions_core_objects():
    text = (API_ROOT / "time-series.md").read_text(encoding="utf-8")
    required = ("TimeSeries", "Workfile", "Equation", "ARIMA", "forecast")
    missing = [item for item in required if item.lower() not in text.lower()]
    assert not missing, "Missing API documentation terms: " + ", ".join(missing)
