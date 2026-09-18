from __future__ import annotations

import copy
import datetime
import threading
from unittest.mock import Mock, patch

import pytest
import srt

from manim import Manager, Scene, tempconfig
from manim._config.output import OutputFormat
from manim.animation.animation import Wait
from manim.constants import RendererType
from manim.renderer.protocol import RendererCapabilities
from manim.scene.scene import SceneInteractRerun
from manim.utils.exceptions import EndSceneEarlyException, RerunSceneException


def test_manager_attaches_to_existing_scene(dry_run):
    scene = Scene()

    manager = Manager(scene)

    assert manager.scene is scene
    assert scene.manager is manager


def test_manager_exposes_the_session_output_snapshot(config):
    config.dry_run = False
    config.format = "gif"
    config.preview = True
    scene = Scene()
    manager = Manager(scene)

    config.format = "none"
    config.preview = False
    config.dry_run = True

    assert manager.output_spec.format is OutputFormat.GIF
    assert manager.output_spec is scene.renderer.file_writer.output_spec
    assert manager.session_spec.presentation.open_after_render is True
    assert manager.session_spec.dry_run is False


def test_manager_exposes_dry_run_session_intent(config):
    config.format = "gif"
    config.dry_run = True
    scene = Scene()
    manager = Manager(scene)

    config.dry_run = False

    assert manager.output_spec.format is OutputFormat.NONE
    assert manager.session_spec.dry_run is True


def test_post_render_preview_requires_an_artifact(dry_run):
    scene = Scene()

    with pytest.raises(ValueError, match="requires a media artifact"):
        Manager(scene).render(preview=True)


@pytest.mark.parametrize("output_format", ["mp4", "mov", "webm", "gif"])
def test_manager_rejects_explicit_video_for_scene_without_play_calls(
    config,
    output_format,
):
    config.format = output_format
    renderer = Mock()
    renderer.capabilities = RendererCapabilities()
    renderer.num_plays = 0
    scene = Scene(renderer)
    manager = Manager(scene)

    with pytest.raises(
        RuntimeError,
        match=f"explicitly requested {output_format.upper()}",
    ):
        manager.post_construct()

    renderer.scene_finished.assert_not_called()


def test_manager_warns_when_automatic_video_falls_back_to_still(config):
    config.format = "auto"
    renderer = Mock()
    renderer.capabilities = RendererCapabilities()
    renderer.num_plays = 0
    scene = Scene(renderer)
    renderer.file_writer.final_file_path = "scene.png"
    manager = Manager(scene)

    with patch("manim.manager.logger.warning") as warning:
        manager.post_construct()

    renderer.scene_finished.assert_called_once_with(scene)
    warning.assert_called_once_with(
        f"{scene} has no play calls. Automatic video output has been saved as a "
        "PNG instead.",
    )


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
    file_writer = Mock()
    file_writer.subcaptions = []
    monkeypatch.setattr(scene.renderer, "play", renderer_play)
    scene.renderer.file_writer = file_writer
    animation = Wait()

    scene.play(animation, run_time=2)

    assert isinstance(scene.manager, Manager)
    renderer_play.assert_called_once_with(scene, animation, run_time=2)
    assert file_writer.subcaptions == []


def test_scene_play_adds_subcaption_with_explicit_duration(dry_run, monkeypatch):
    scene = Scene()
    file_writer = Mock()
    file_writer.subcaptions = []
    scene.renderer.file_writer = file_writer
    scene.renderer.time = 2.5
    renderer_play = Mock(
        side_effect=lambda *args, **kwargs: setattr(scene.renderer, "time", 4.5)
    )
    monkeypatch.setattr(scene.renderer, "play", renderer_play)
    animation = Wait()

    scene.play(
        animation,
        run_time=2,
        subcaption="Hello",
        subcaption_duration=1.25,
        subcaption_offset=0.25,
    )

    renderer_play.assert_called_once_with(scene, animation, run_time=2)
    assert file_writer.subcaptions == [
        srt.Subtitle(
            index=0,
            content="Hello",
            start=datetime.timedelta(seconds=2.75),
            end=datetime.timedelta(seconds=4.0),
        )
    ]


def test_scene_play_preserves_scene_add_subcaption_override(dry_run, monkeypatch):
    subcaptions: list[tuple[str, float, float]] = []

    class CustomSubcaptionScene(Scene):
        def add_subcaption(
            self, content: str, duration: float = 1, offset: float = 0
        ) -> None:
            subcaptions.append((content, duration, offset))

    scene = CustomSubcaptionScene()
    scene.renderer.time = 2.5
    renderer_play = Mock(
        side_effect=lambda *args, **kwargs: setattr(scene.renderer, "time", 4.5)
    )
    monkeypatch.setattr(scene.renderer, "play", renderer_play)

    scene.play(
        Wait(),
        subcaption="Hello",
        subcaption_duration=1.25,
        subcaption_offset=0.25,
    )

    assert subcaptions == [("Hello", 1.25, -1.75)]


def test_scene_play_uses_animation_duration_for_default_subcaption_after_skip(
    dry_run, monkeypatch
):
    scene = Scene()
    file_writer = Mock()
    file_writer.subcaptions = []
    scene.renderer.file_writer = file_writer
    scene.renderer.time = 1.0
    scene.renderer.skip_animations = True
    renderer_play = Mock(
        side_effect=lambda *args, **kwargs: setattr(scene.renderer, "time", 3.5)
    )
    monkeypatch.setattr(scene.renderer, "play", renderer_play)

    scene.play(Wait(), subcaption="Hello")

    assert file_writer.subcaptions == [
        srt.Subtitle(
            index=0,
            content="Hello",
            start=datetime.timedelta(seconds=1.0),
            end=datetime.timedelta(seconds=3.5),
        )
    ]


def test_scene_play_queues_interactive_calls_without_attaching_manager(
    dry_run, monkeypatch
):
    scene = Scene()
    scene.interactive_mode = True
    renderer_play = Mock()
    monkeypatch.setattr(scene.renderer, "play", renderer_play)
    worker_thread = threading.Thread(name="Worker")
    monkeypatch.setattr(threading, "current_thread", lambda: worker_thread)

    with tempconfig({"renderer": RendererType.OPENGL}):
        scene.play(
            Wait(),
            run_time=2,
            subcaption="Hello",
            subcaption_duration=1.25,
            subcaption_offset=0.5,
        )

    assert scene.manager is None
    renderer_play.assert_not_called()
    queued_call = scene.queue.get_nowait()
    assert isinstance(queued_call, SceneInteractRerun)
    assert queued_call.sender == "play"
    assert queued_call.kwargs == {
        "run_time": 2,
        "subcaption": "Hello",
        "subcaption_duration": 1.25,
        "subcaption_offset": 0.5,
    }


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
