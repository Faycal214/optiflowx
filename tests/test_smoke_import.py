def test_import_optiflowx():
    """Simple smoke test: ensure package imports and exposes __version__."""
    import importlib

    pkg = importlib.import_module("optiflowx")
    assert hasattr(pkg, "__version__"), "optiflowx should expose __version__"
    assert isinstance(pkg.__version__, str) and pkg.__version__, "__version__ should be a non-empty string"
