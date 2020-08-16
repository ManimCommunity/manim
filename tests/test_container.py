import pytest
from manim import Container, Mobject, Scene


def test_ABC():
    """Test that the Container class cannot be instantiated."""
    with pytest.raises(TypeError):
        Container()

    # The following should work without raising exceptions
    Mobject()
    Scene()


def container_add(obj, get_submobjects):
    """Call this function with a Container instance to test its add() method."""
    # check that obj.submobjects is updated correctly
    assert len(get_submobjects()) == 0
    obj.add(Mobject())
    assert len(get_submobjects()) == 1
    obj.add(*(Mobject() for _ in range(10)))
    assert len(get_submobjects()) == 11

    # check that adding a mobject twice does not actually add it twice
    repeated = Mobject()
    obj.add(repeated)
    assert len(get_submobjects()) == 12
    obj.add(repeated)
    assert len(get_submobjects()) == 12

    # check that Container.add() returns the Mobject (for chained calls)
    assert obj.add(Mobject()) is obj


def container_remove(obj, get_submobjects):
    """Call this function with a Container instance to test its remove() method."""
    to_remove = Mobject()
    obj.add(to_remove)
    obj.add(*(Mobject() for _ in range(10)))
    assert len(get_submobjects()) == 11
    obj.remove(to_remove)
    assert len(get_submobjects()) == 10
    obj.remove(to_remove)
    assert len(get_submobjects()) == 10

    # check that Container.remove() returns the instance (for chained calls)
    assert obj.add(Mobject()) is obj


def test_mobject_add():
    """Test Mobject.add()."""
    obj = Mobject()
    container_add(obj, lambda: obj.submobjects)

    # a Mobject cannot contain itself
    with pytest.raises(ValueError):
        obj.add(obj)


def test_mobject_remove():
    """Test Mobject.remove()."""
    obj = Mobject()
    container_remove(obj, lambda: obj.submobjects)


def test_scene_add():
    """Test Scene.add()."""
    scene = Scene()
    container_add(scene, lambda: scene.mobjects)


def test_scene_remove():
    """Test Scene.remove()."""
    scene = Scene()
    container_remove(scene, lambda: scene.mobjects)
