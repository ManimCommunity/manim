from __future__ import annotations

import copy
from unittest.mock import Mock

import pytest

from manim import Manager, Scene
from manim.animation.animation import Wait
from manim.utils.exceptions import RerunSceneException


def test_manager_attaches_to_existing_scene(dry_run):
    scene = Scene()

    manager = Manager(scene)

    assert manager.scene is scene
    assert scene.manager is manager


def test_manager_rejects_second_attachment(dry_run):
    scene = Scene()
    manager = Manager(scene)

    with pytest.raises(ValueError, match="already attached"):
        Manager(scene)

    assert scene.manager is manager


def test_deepcopy_leaves_clone_managerless(dry_run):
    scene = Scene()
    manager = Manager(scene)
    scene.queue = None

    clone = copy.deepcopy(scene)

    assert scene.manager is manager
    assert clone.manager is None


def test_scene_render_lazily_creates_manager(dry_run):
    lifecycle: list[str] = []

    class LifecycleScene(Scene):
        def setup(self):
            lifecycle.append("setup")

        def construct(self):
            lifecycle.append("construct")

        def tear_down(self):
            lifecycle.append("tear_down")

    scene = LifecycleScene()
    assert scene.manager is None

    assert scene.render() is False

    assert isinstance(scene.manager, Manager)
    assert lifecycle == ["setup", "construct", "tear_down"]


def test_cli_style_render_preserves_scene_render_override(dry_run):
    calls: list[bool] = []

    class RenderOverrideScene(Scene):
        def render(self, preview: bool = False) -> bool:
            calls.append(preview)
            return True

    scene = RenderOverrideScene()
    Manager(scene)

    assert scene.render(preview=True) is True
    assert calls == [True]


def test_manager_render_uses_scene_lifecycle(dry_run):
    lifecycle: list[str] = []

    class LifecycleScene(Scene):
        def setup(self):
            lifecycle.append("setup")

        def construct(self):
            lifecycle.append("construct")

        def tear_down(self):
            lifecycle.append("tear_down")

    scene = LifecycleScene()

    assert Manager(scene).render() is False

    assert lifecycle == ["setup", "construct", "tear_down"]


def test_manager_render_handles_rerun_without_finishing_scene(dry_run):
    lifecycle: list[str] = []
    renderer = Mock()
    renderer.num_plays = 3

    class RerunScene(Scene):
        def construct(self):
            raise RerunSceneException

        def tear_down(self):
            lifecycle.append("tear_down")

    scene = RerunScene(renderer)

    assert Manager(scene).render() is True

    renderer.clear_screen.assert_called_once_with()
    assert renderer.num_plays == 0
    renderer.scene_finished.assert_not_called()
    assert lifecycle == []


def test_scene_play_forwards_through_manager(dry_run, monkeypatch):
    scene = Scene()
    renderer_play = Mock()
    monkeypatch.setattr(scene.renderer, "play", renderer_play)
    animation = Wait()

    scene.play(animation, run_time=2)

    assert isinstance(scene.manager, Manager)
    renderer_play.assert_called_once_with(scene, animation, run_time=2)
