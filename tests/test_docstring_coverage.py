import inspect
import optiflowx.stochastic as stochastic


def test_public_classes_and_methods_have_standard_docstrings():
    for name in stochastic.__all__:
        obj = getattr(stochastic, name)
        if not isinstance(obj, type):
            continue
        assert obj.__doc__, f"Missing class docstring: {obj.__module__}.{obj.__name__}"
        if obj.__module__.endswith(".exceptions"):
            continue
        assert "Mathematical object" in obj.__doc__
        assert "Course basis" in obj.__doc__
        assert "Examples" in obj.__doc__
        try:
            sig = inspect.signature(obj.__init__)
            has_params = any(
                p.name not in {"self", "cls"} and not p.name.startswith("_")
                for p in sig.parameters.values()
            )
        except (TypeError, ValueError):
            has_params = False
        if has_params:
            assert "Parameters" in obj.__doc__, obj.__name__
        for method_name, member in inspect.getmembers(obj):
            if method_name.startswith("_") and method_name != "__init__":
                continue
            if method_name == "__init__" and hasattr(obj, "__dataclass_fields__"):
                continue
            func = member.fget if isinstance(member, property) else member
            if inspect.isfunction(func):
                assert func.__doc__, f"Missing method docstring: {obj.__name__}.{method_name}"
                assert "Examples" in func.__doc__, f"Missing Examples section: {obj.__name__}.{method_name}"
