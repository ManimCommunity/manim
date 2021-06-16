import pytest

from manim import Mobject


def test_mobject_add():
    """Test Mobject.add()."""
    """Call this function with a Container instance to test its add() method."""
    # check that obj.submobjects is updated correctly
    obj = Mobject()
    assert len(obj.submobjects) == 0
    obj.add(Mobject())
    assert len(obj.submobjects) == 1
    obj.add(*(Mobject() for _ in range(10)))
    assert len(obj.submobjects) == 11

    # check that adding a mobject twice does not actually add it twice
    repeated = Mobject()
    obj.add(repeated)
    assert len(obj.submobjects) == 12
    obj.add(repeated)
    assert len(obj.submobjects) == 12

    # check that Mobject.add() returns the Mobject (for chained calls)
    assert obj.add(Mobject()) is obj
    obj = Mobject()

    # a Mobject cannot contain itself
    with pytest.raises(ValueError):
        obj.add(obj)

    # can only add Mobjects
    with pytest.raises(TypeError):
        obj.add("foo")


def test_mobject_remove():
    """Test Mobject.remove()."""
    obj = Mobject()
    to_remove = Mobject()
    obj.add(to_remove)
    obj.add(*(Mobject() for _ in range(10)))
    assert len(obj.submobjects) == 11
    obj.remove(to_remove)
    assert len(obj.submobjects) == 10
    obj.remove(to_remove)
    assert len(obj.submobjects) == 10

    assert obj.remove(Mobject()) is obj
