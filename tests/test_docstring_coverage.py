import importlib
import inspect
import pkgutil

import stochx.timeseries as timeseries


def _public_modules():
    return (
        timeseries.__name__,
        *(
            info.name
            for info in pkgutil.iter_modules(
                timeseries.__path__, prefix=f"{timeseries.__name__}."
            )
            if not info.name.rsplit(".", 1)[-1].startswith("_")
        ),
    )


def _public_objects(module):
    for name, obj in vars(module).items():
        if name.startswith("_"):
            continue
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        if inspect.isclass(obj) or inspect.isfunction(obj):
            yield name, obj


def test_public_timeseries_api_docstrings_exist():
    missing = []
    for module_name in _public_modules():
        module = importlib.import_module(module_name)
        for name, obj in _public_objects(module):
            if not obj.__doc__:
                missing.append(f"{module_name}.{name}")
    assert not missing, "Missing docstrings: " + ", ".join(sorted(missing))
