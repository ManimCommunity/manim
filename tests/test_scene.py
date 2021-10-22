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

        # Check that adding a mobject twice does not actually add it twice
        repeated = Mobject()
        scene.add(repeated)
        assert len(scene.mobjects) == 12
        scene.add(repeated)
        assert len(scene.mobjects) == 12

        # Check that Scene.add() returns the Scene (for chained calls)
        assert scene.add(Mobject()) is scene
        to_remove = Mobject()
        scene = Scene()
        scene.add(to_remove)
        scene.add(*(Mobject() for _ in range(10)))
        assert len(scene.mobjects) == 11
        scene.remove(to_remove)
        assert len(scene.mobjects) == 10
        scene.remove(to_remove)
        assert len(scene.mobjects) == 10

        # Check that Scene.remove() returns the instance (for chained calls)
        assert scene.add(Mobject()) is scene
