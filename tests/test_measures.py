import pytest

from stereotype_bench.measures import get_measure, list_measures


def test_stub_registered():
    assert "stub" in list_measures()


def test_stub_returns_zero():
    m = get_measure("stub")
    assert m.score("anything") == 0.0
    assert m.score("a much longer piece of text with many tokens") == 0.0


def test_unknown_measure_raises():
    with pytest.raises(KeyError):
        get_measure("nonexistent-measure")


def test_axis_labels_are_tuple_of_two():
    m = get_measure("stub")
    assert isinstance(m.axis_labels, tuple)
    assert len(m.axis_labels) == 2
