from optiflowx.core import SearchSpace


def test_sample_returns_dict():
    s = SearchSpace()
    s.add("a", "continuous", (0.1, 1.0))
    p = s.sample()
    assert isinstance(p, dict)
    assert "a" in p
