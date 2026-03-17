from __future__ import annotations

import datetime

import pytest

from manim import Circle, Dot, FadeIn, Group, Mobject, Scene, Square
from manim.animation.animation import Wait


def test_scene_add_remove(dry_run):
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


def test_scene_time(dry_run):
    scene = Scene()
    assert scene.time == 0
    scene.wait(2)
    assert scene.time == 2
    scene.play(FadeIn(Circle()), run_time=0.5)
    assert pytest.approx(scene.time) == 2.5
    scene.renderer._original_skipping_status = True
    scene.play(FadeIn(Square()), run_time=5)  # this animation gets skipped.
    assert pytest.approx(scene.time) == 7.5


def test_subcaption(dry_run):
    scene = Scene()
    scene.add_subcaption("Testing add_subcaption", duration=1, offset=0)
    scene.wait()
    scene.play(
        Wait(),
        run_time=2,
        subcaption="Testing Scene.play subcaption interface",
        subcaption_duration=1.5,
        subcaption_offset=0.5,
    )
    subcaptions = scene.renderer.file_writer.subcaptions
    assert len(subcaptions) == 2
    assert subcaptions[0].start == datetime.timedelta(seconds=0)
    assert subcaptions[0].end == datetime.timedelta(seconds=1)
    assert subcaptions[0].content == "Testing add_subcaption"
    assert subcaptions[1].start == datetime.timedelta(seconds=1.5)
    assert subcaptions[1].end == datetime.timedelta(seconds=3)
    assert subcaptions[1].content == "Testing Scene.play subcaption interface"


def test_replace(dry_run):
    def assert_names(mobjs, names):
        assert len(mobjs) == len(names)
        for i in range(0, len(mobjs)):
            assert mobjs[i].name == names[i]

    scene = Scene()

    first = Mobject(name="first")
    second = Mobject(name="second")
    third = Mobject(name="third")
    fourth = Mobject(name="fourth")

    scene.add(first)
    scene.add(Group(second, third, name="group"))
    scene.add(fourth)
    assert_names(scene.mobjects, ["first", "group", "fourth"])
    assert_names(scene.mobjects[1], ["second", "third"])

    alpha = Mobject(name="alpha")
    beta = Mobject(name="beta")

    scene.replace(first, alpha)
    assert_names(scene.mobjects, ["alpha", "group", "fourth"])
    assert_names(scene.mobjects[1], ["second", "third"])

    scene.replace(second, beta)
    assert_names(scene.mobjects, ["alpha", "group", "fourth"])
    assert_names(scene.mobjects[1], ["beta", "third"])


def test_reproducible_scene(dry_run):
    import numpy as np

    scene = Scene(random_seed=42)
    dots1 = []
    for _ in range(10):
        dot = Dot(np.random.uniform(-3, 3, size=3))  # noqa: NPY002
        dots1.append(dot)
    scene.add(*dots1)

    scene2 = Scene(random_seed=42)
    dots2 = []
    for _ in range(5):
        dot = Dot(np.random.uniform(-3, 3, size=3))  # noqa: NPY002
        dots2.append(dot)
    scene2.add(*dots2)

    for d1, d2 in zip(dots1, dots2, strict=False):
        np.testing.assert_allclose(d1.get_center(), d2.get_center())


def test_random_color_reproducibility_with_seed(dry_run):
    from manim import random_color, tempconfig

    with tempconfig({"seed": 123}):
        # First run: create scene (which seeds global random state) and generate colors
        scene1 = Scene()
        colors_first_run = [random_color() for _ in range(5)]

        # Interrupt with a scene that has an explicit different seed
        scene_explicit = Scene(random_seed=999)
        colors_interrupted = [random_color() for _ in range(3)]

        # Second run: create a new scene without explicit seed (should use config.seed)
        scene2 = Scene()
        colors_second_run = [random_color() for _ in range(5)]

        # The colors from the first and second run should match
        # because both scenes were seeded with config.seed=123
        assert colors_first_run == colors_second_run

        # The interrupted colors should be different (seeded with 999)
        assert colors_interrupted != colors_first_run[:3]
