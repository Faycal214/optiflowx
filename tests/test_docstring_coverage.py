import importlib
import inspect
import pkgutil

import optiflowx.stochastic as stochastic


PUBLIC_STOCHASTIC_MODULES = tuple(
    info.name
    for info in pkgutil.iter_modules(stochastic.__path__, prefix=f"{stochastic.__name__}.")
    if not info.name.rsplit(".", 1)[-1].startswith("_")
)


def _public_objects(module):
    """Return public classes and functions defined by ``module`` itself."""
    for name, obj in vars(module).items():
        if name.startswith("_"):
            continue
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        if inspect.isclass(obj) or inspect.isfunction(obj):
            yield name, obj


def _assert_standard_class_docstring(cls):
    assert cls.__doc__, f"Missing class docstring: {cls.__module__}.{cls.__name__}"
    if cls.__module__.endswith(".exceptions"):
        return

    assert "Mathematical object" in cls.__doc__
    assert "Course basis" in cls.__doc__
    assert "Examples" in cls.__doc__

    try:
        signature = inspect.signature(cls.__init__)
        has_public_parameters = any(
            parameter.name not in {"self", "cls"}
            and not parameter.name.startswith("_")
            for parameter in signature.parameters.values()
        )
    except (TypeError, ValueError):
        has_public_parameters = False

    if has_public_parameters:
        assert "Parameters" in cls.__doc__, cls.__name__

    for method_name in dir(cls):
        if method_name.startswith("_") and method_name != "__init__":
            continue
        if method_name == "__init__" and hasattr(cls, "__dataclass_fields__"):
            continue

        raw_member = inspect.getattr_static(cls, method_name)
        if isinstance(raw_member, (classmethod, staticmethod)):
            func = raw_member.__func__
        elif isinstance(raw_member, property):
            func = raw_member.fget
        else:
            func = raw_member

        if func is None or not inspect.isfunction(func):
            continue

        assert func.__doc__, f"Missing method docstring: {cls.__name__}.{method_name}"
        assert "Examples" in func.__doc__, (
            f"Missing Examples section: {cls.__name__}.{method_name}"
        )


def _assert_standard_function_docstring(function):
    assert function.__doc__, (
        f"Missing function docstring: {function.__module__}.{function.__name__}"
    )
    assert "Examples" in function.__doc__, (
        f"Missing Examples section: {function.__module__}.{function.__name__}"
    )


def test_public_api_docstrings_are_discovered_independently_of___all__():
    """Cover public classes/functions from the canonical stochastic modules."""
    for module_name in PUBLIC_STOCHASTIC_MODULES:
        module = importlib.import_module(module_name)
        for _, obj in _public_objects(module):
            if inspect.isclass(obj):
                _assert_standard_class_docstring(obj)
            elif inspect.isfunction(obj):
                _assert_standard_function_docstring(obj)
