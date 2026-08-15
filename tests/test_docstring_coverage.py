import importlib
import inspect
import pkgutil

import optiflowx.stochastic as stochastic


PUBLIC_STOCHASTIC_MODULES = tuple(
    info.name
    for info in pkgutil.iter_modules(
        stochastic.__path__, prefix=f"{stochastic.__name__}."
    )
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


def _assert_documented_function(func, owner, *, require_examples=True):
    assert func.__doc__, f"Missing docstring: {owner}"
    if require_examples:
        assert "Examples" in func.__doc__, f"Missing Examples section: {owner}"


def _assert_standard_property_docstring(prop, owner):
    if prop.fget is not None:
        _assert_documented_function(prop.fget, f"{owner}.getter")
    if prop.fset is not None:
        _assert_documented_function(
            prop.fset, f"{owner}.setter", require_examples=False
        )
    if prop.fdel is not None:
        _assert_documented_function(
            prop.fdel, f"{owner}.deleter", require_examples=False
        )


def _assert_standard_descriptor_docstring(descriptor, owner):
    """Require documentation for custom public descriptors.

    Built-in descriptor categories with specialized handling are excluded.
    Any other object implementing ``__get__`` is treated as a public custom
    descriptor and must provide its own docstring.
    """
    if isinstance(descriptor, (property, classmethod, staticmethod)):
        return
    if hasattr(descriptor, "__get__"):
        assert descriptor.__doc__, f"Missing descriptor docstring: {owner}"


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
        owner = f"{cls.__name__}.{method_name}"

        if isinstance(raw_member, property):
            _assert_standard_property_docstring(raw_member, owner)
            continue

        if isinstance(raw_member, (classmethod, staticmethod)):
            _assert_documented_function(raw_member.__func__, owner)
            continue

        if inspect.isfunction(raw_member):
            _assert_documented_function(raw_member, owner)
            continue

        _assert_standard_descriptor_docstring(raw_member, owner)


def _assert_standard_function_docstring(function):
    _assert_documented_function(
        function,
        f"{function.__module__}.{function.__name__}",
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


def test_property_accessors_are_covered_by_the_docstring_guard():
    """Ensure public property getters, setters, and deleters are validated."""

    class Fixture:
        @property
        def value(self):
            """Getter documentation.

            Examples
            --------
            ``fixture.value``
            """
            return 1

        @value.setter
        def value(self, new_value):
            """Setter documentation."""
            self._value = new_value

        @value.deleter
        def value(self):
            """Deleter documentation."""
            del self._value

    _assert_standard_class_docstring.__call__ if False else None
    _assert_standard_property_docstring(Fixture.__dict__["value"], "Fixture.value")


def test_custom_public_descriptors_are_not_silently_ignored():
    """Ensure a custom descriptor without documentation fails the contract."""

    class DocumentedDescriptor:
        """A documented descriptor used by the coverage fixture."""

        def __get__(self, instance, owner):
            return self

    class UndocumentedDescriptor:
        def __get__(self, instance, owner):
            return self

    _assert_standard_descriptor_docstring(
        DocumentedDescriptor(), "DocumentedDescriptor"
    )
    try:
        _assert_standard_descriptor_docstring(
            UndocumentedDescriptor(), "UndocumentedDescriptor"
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("Undocumented custom descriptor was not rejected")
