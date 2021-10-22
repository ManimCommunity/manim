import pytest

from manim import Circle, FadeIn, Mobject, Scene, Square, config, tempconfig


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


def test_scene_time():
    with tempconfig({"dry_run": True}):
        scene = Scene()
        assert scene.renderer.time == 0
        scene.wait(2)
        assert scene.renderer.time == 2
        scene.play(FadeIn(Circle()), run_time=0.5)
        assert pytest.approx(scene.renderer.time) == 2.5
        scene.renderer._original_skipping_status = True
        scene.play(FadeIn(Square()), run_time=5)  # this animation gets skipped.
        assert pytest.approx(scene.renderer.time) == 7.5
