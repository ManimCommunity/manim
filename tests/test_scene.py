import pytest

from manim import Mobject, Scene, tempconfig


def test_scene_add_remove():
    with tempconfig({"dry_run": True}):
        scene = Scene()
        assert len(scene.mobjects) == 0
        scene.add(Mobject())
        assert len(scene.mobjects) == 1
        scene.add(*(Mobject() for _ in range(10)))
        assert len(scene.mobjects) == 11

        # check that adding a mobject twice does not actually add it twice
        repeated = Mobject()
        scene.add(repeated)
        assert len(scene.mobjects) == 12
        scene.add(repeated)
        assert len(scene.mobjects) == 12

        # check that Scene.add() returns the Scene (for chained calls)
        assert scene.add(Mobject()) is scene

        # can only add Mobjects
        with pytest.raises(TypeError):
            scene.add("foo")

        assert scene.add(Mobject()) is scene

    scene = Scene()
    to_remove = Mobject()
    scene.add(to_remove)
    scene.add(*(Mobject() for _ in range(10)))
    assert len(scene.mobjects) == 11
    scene.remove(to_remove)
    assert len(scene.mobjects) == 10
    scene.remove(to_remove)
    assert len(scene.mobjects) == 10
