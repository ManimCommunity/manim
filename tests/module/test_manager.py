from __future__ import annotations

import copy
import datetime
from unittest.mock import Mock

import pytest
import srt

from manim import Manager, Scene
from manim.animation.animation import Wait
from manim.utils.exceptions import EndSceneEarlyException, RerunSceneException


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


def test_manager_render_uses_lifecycle_hooks(dry_run):
    hooks: list[str] = []

    class HookManager(Manager[Scene]):
        def setup(self):
            hooks.append("setup")
            super().setup()

        def construct(self):
            hooks.append("construct")
            super().construct()

        def tear_down(self):
            hooks.append("tear_down")
            super().tear_down()

        def post_construct(self):
            hooks.append("post_construct")
            super().post_construct()

    assert HookManager(Scene()).render() is False

    assert hooks == ["setup", "construct", "tear_down", "post_construct"]


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


def test_manager_render_finalizes_after_early_scene_end(dry_run):
    lifecycle: list[str] = []
    renderer = Mock()
    renderer.num_plays = 0
    renderer.scene_finished.side_effect = lambda scene: lifecycle.append(
        "post_construct"
    )

    class EarlyEndScene(Scene):
        def construct(self):
            raise EndSceneEarlyException

        def tear_down(self):
            lifecycle.append("tear_down")

    scene = EarlyEndScene(renderer)

    assert Manager(scene).render() is False

    renderer.scene_finished.assert_called_once_with(scene)
    assert lifecycle == ["tear_down", "post_construct"]


def test_scene_play_forwards_through_manager(dry_run, monkeypatch):
    scene = Scene()
    renderer_play = Mock()
    monkeypatch.setattr(scene.renderer, "play", renderer_play)
    animation = Wait()

    scene.play(animation, run_time=2)

    assert isinstance(scene.manager, Manager)
    renderer_play.assert_called_once_with(scene, animation, run_time=2)


def test_manager_views_follow_the_scene_renderer(dry_run):
    scene = Scene()
    manager = Manager(scene)
    replacement_renderer = Mock()
    replacement_renderer.camera = Mock()
    replacement_renderer.file_writer = Mock()
    replacement_renderer.time = 1.5
    replacement_renderer.num_plays = 2
    replacement_renderer.skip_animations = False

    scene.renderer = replacement_renderer

    assert manager.renderer is replacement_renderer
    assert manager.camera is replacement_renderer.camera
    assert manager.file_writer is replacement_renderer.file_writer
    assert manager.time == 1.5
    assert manager.num_plays == 2
    assert manager.skip_animations is False

    manager.time = 2.5
    manager.num_plays = 3
    manager.skip_animations = True

    assert replacement_renderer.time == 2.5
    assert replacement_renderer.num_plays == 3
    assert replacement_renderer.skip_animations is True


def test_scene_next_section_delegates_to_manager(dry_run):
    scene = Scene()
    file_writer = Mock()
    scene.renderer.file_writer = file_writer

    section_type = "presentation.skip"

    scene.next_section("intro", section_type, skip_animations=True)

    assert isinstance(scene.manager, Manager)
    file_writer.next_section.assert_called_once_with("intro", section_type, True)


def test_scene_add_subcaption_delegates_to_manager(dry_run):
    scene = Scene()
    file_writer = Mock()
    file_writer.subcaptions = []
    scene.renderer.file_writer = file_writer
    scene.renderer.time = 2.5

    scene.add_subcaption("Hello", duration=1.5, offset=0.25)

    assert isinstance(scene.manager, Manager)
    assert file_writer.subcaptions == [
        srt.Subtitle(
            index=0,
            content="Hello",
            start=datetime.timedelta(seconds=2.75),
            end=datetime.timedelta(seconds=4.25),
        )
    ]


def test_scene_add_sound_delegates_to_manager_and_honors_skip_state(dry_run):
    scene = Scene()
    file_writer = Mock()
    scene.renderer.file_writer = file_writer
    scene.renderer.time = 2.5

    scene.add_sound("bell.wav", time_offset=0.25, gain=-3, marker="test")

    assert isinstance(scene.manager, Manager)
    file_writer.add_sound.assert_called_once_with("bell.wav", 2.75, -3, marker="test")

    file_writer.add_sound.reset_mock()
    scene.renderer.skip_animations = True

    scene.add_sound("skipped.wav")

    file_writer.add_sound.assert_not_called()
