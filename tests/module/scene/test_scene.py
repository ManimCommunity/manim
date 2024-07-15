from __future__ import annotations

import datetime

import pytest

from manim import Circle, FadeIn, Group, Manager, Mobject, Scene, Square
from manim.animation.animation import Wait


def test_scene_add_remove(dry_run):
    manager = Manager(Scene)
    scene = manager.scene
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
    manager = Manager(Scene)
    scene = manager.scene
    assert scene.time == 0
    scene.wait(2)
    assert scene.time == 2
    scene.play(FadeIn(Circle()), run_time=0.5)
    assert pytest.approx(scene.time) == 2.5
    scene._original_skipping_status = True
    scene.play(FadeIn(Square()), run_time=5)  # this animation gets skipped.
    assert pytest.approx(scene.renderer.time) == 7.5


def test_subcaption(dry_run):
    manager = Manager(Scene)
    scene = manager.scene
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

    manager = Manager(Scene)
    scene = manager.scene

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
