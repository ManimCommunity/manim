"""Evaluation runs normal animation steps without drawing or writing media."""

from unittest.mock import Mock

import numpy as np
import pytest

from manim import (
    Animation,
    Manager,
    Scene,
    Square,
    ThreeDScene,
    tempconfig,
)


class TraceScene(Scene):
    def setup(self):
        self.trace = [("setup", self.time)]

    def construct(self):
        scene = self

        class Probe(Animation):
            def begin(self):
                scene.trace.append(("begin", scene.time))
                super().begin()

            def interpolate_mobject(self, alpha):
                scene.trace.append(("interpolate", scene.time, float(alpha)))
                self.mobject.set_x(scene.time)

            def finish(self):
                scene.trace.append(("finish", scene.time))
                super().finish()

            def clean_up_from_scene(self, scene):
                scene.trace.append(("cleanup", scene.time))
                super().clean_up_from_scene(scene)

        self.add_updater(lambda dt: self.trace.append(("update", self.time, float(dt))))
        self.play(Probe(Square(), run_time=0.3), subcaption="first")
        self.wait(0.3, frozen_frame=True)
        start = self.time
        self.wait(1, stop_condition=lambda: self.time >= start + 0.5)

    def tear_down(self):
        self.trace.append(("teardown", self.time))


@pytest.fixture(params=["cairo", "opengl"])
def settings(request, tmp_path):
    with tempconfig(
        {
            "renderer": request.param,
            "format": "none",
            "frame_rate": 4,
            "pixel_width": 32,
            "pixel_height": 16,
            "live_preview": False,
            "disable_caching": True,
            "progress_bar": "none",
            "media_dir": str(tmp_path),
        }
    ):
        yield tmp_path


def test_real_evaluation_matches_uncached_render(settings, monkeypatch):
    rendered = TraceScene()
    rendered.render()
    evaluated = TraceScene()
    manager = Manager(evaluated)
    reject = Mock(side_effect=AssertionError("raster/output policy executed"))
    monkeypatch.setattr(evaluated.renderer, "_start_animation", reject)
    monkeypatch.setattr(evaluated.renderer, "_prepare_animation", reject)
    monkeypatch.setattr(evaluated.renderer, "scene_finished", reject)
    monkeypatch.setattr(evaluated.renderer, "_file_writer_class", reject)
    monkeypatch.setattr(evaluated.renderer, "render", reject)
    monkeypatch.setattr("manim._run.listeners.get_hash_from_play_call", reject)
    # Evaluation still runs every animation with caching and range skipping enabled.
    with tempconfig(
        {
            "disable_caching": False,
            "from_animation_number": 10,
            "upto_animation_number": 0,
        }
    ):
        evaluated.renderer._original_skipping_status = True
        manager.evaluate()
    assert evaluated.trace == rendered.trace
    assert evaluated.time == rendered.time == 1.25
    assert manager.num_plays == 3
    assert manager._file_writer is None
    assert manager._closed
    reject.assert_not_called()


def test_declarations_do_not_open_media_logs_or_decode_audio(settings, monkeypatch):
    class Declarations(Scene):
        def construct(self):
            self.next_section("not skipped", skip_animations=True)
            self.wait(0.3, frozen_frame=False)
            self.add_subcaption("caption", duration=0.5)
            self.add_sound("missing-file.wav", time_offset=0.25, gain=-3)

    with tempconfig(
        {"format": "mp4", "log_to_file": True, "log_dir": str(settings / "logs")}
    ):
        scene = Declarations()
        manager = Manager(scene)
        reject = Mock(side_effect=AssertionError("resource acquired"))
        monkeypatch.setattr(scene.renderer, "_file_writer_class", reject)
        monkeypatch.setattr("manim.manager.set_file_logger", reject)
        if hasattr(scene.renderer, "open"):
            monkeypatch.setattr(scene.renderer, "open", reject)
        else:
            monkeypatch.setattr(scene.renderer, "_get_target", reject)
        manager.evaluate()
        assert scene.time == 0.5
        assert not list(settings.rglob("*"))
        reject.assert_not_called()


@pytest.mark.parametrize("demand", ["writer", "render", "interactive", "mesh"])
def test_unsupported_requests_raise(settings, demand):
    class Unsupported(Scene):
        def construct(self):
            if demand == "writer":
                self.renderer.file_writer
            elif demand == "interactive":
                self.interactive_embed()
            elif demand == "mesh":
                self.meshes.append(Mock())
            else:
                self.render()

    scene = Unsupported()
    manager = Manager(scene)
    with pytest.raises(RuntimeError, match="during evaluation"):
        manager.evaluate()
    assert manager._closed
    assert manager._file_writer is None


def test_images_can_be_requested_during_evaluation(settings):
    class Snapshot(Scene):
        def construct(self):
            self.add(Square(fill_opacity=1))
            self.wait(0.25, frozen_frame=False)
            self.image = self.get_image()

    scene = Snapshot()
    manager = Manager(scene)
    manager.evaluate()
    assert scene.image.size == (32, 16)
    assert manager._file_writer is None


def test_evaluation_failure_preserves_primary_exception(settings):
    error = KeyboardInterrupt("primary")

    class Broken(Scene):
        def construct(self):
            self.wait(0.3, frozen_frame=False)
            raise error

    manager = Manager(Broken())
    with pytest.raises(KeyboardInterrupt) as caught:
        manager.evaluate()
    assert caught.value is error
    assert manager._closed
    assert manager._file_writer is None
    assert not manager._evaluating


def test_images_after_evaluation_do_not_advance_time(settings):
    scene = TraceScene()
    manager = Manager(scene)
    manager.evaluate()
    assert scene.get_image().size == (32, 16)
    assert scene.time == 1.25
    assert manager._file_writer is None


def test_a_scene_can_be_evaluated_once(settings):
    with Manager(Scene()) as manager:
        manager.evaluate()
        with pytest.raises(RuntimeError, match="unused Scene"):
            manager.evaluate()


def test_camera_queries_match_a_render(tmp_path):
    class Projection(ThreeDScene):
        def construct(self):
            self.observed = []
            self.camera.set_theta(0)

            def update(dt):
                self.camera.set_theta(self.time)
                self.observed.append(
                    self.camera.project_point(np.array([1.0, 0.0, 0.0]))
                )

            self.add_updater(update)
            self.wait(1, frozen_frame=False)

    with tempconfig(
        {
            "renderer": "cairo",
            "format": "none",
            "frame_rate": 4,
            "pixel_width": 32,
            "pixel_height": 16,
            "disable_caching": True,
            "progress_bar": "none",
            "media_dir": str(tmp_path),
        }
    ):
        ordinary = Projection()
        ordinary.render()
        evaluated = Projection()
        Manager(evaluated).evaluate()
        np.testing.assert_array_equal(ordinary.observed, evaluated.observed)
