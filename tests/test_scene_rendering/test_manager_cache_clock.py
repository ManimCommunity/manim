"""Real segment reuse must preserve the engine time visible to user code."""

from unittest.mock import Mock

import av
import numpy as np
import pytest

from manim import Animation, Scene, Square, tempconfig


class TimeDriven(Animation):
    def __init__(self, mobject, scene):
        self.scene = scene
        super().__init__(mobject, run_time=0.3)

    def interpolate_mobject(self, alpha):
        self.mobject.set_x(self.scene.time)


class TimeDrivenScene(Scene):
    def construct(self):
        for _ in range(4):
            square = Square(side_length=0.5)
            self.add(square)
            self.play(TimeDriven(square, self))
            self.remove(square)


def decoded_frames(path):
    with av.open(str(path)) as video:
        return [frame.to_ndarray(format="rgba") for frame in video.decode(video=0)]


@pytest.mark.parametrize("backend", ["cairo", "opengl"])
def test_time_dependent_cache_and_fractional_downstream_epochs(tmp_path, backend):
    with tempconfig(
        {
            "renderer": backend,
            "format": "mp4",
            "media_dir": str(tmp_path),
            "frame_rate": 4,
            "pixel_width": 64,
            "pixel_height": 32,
            "live_preview": False,
            "disable_caching": False,
            "progress_bar": "none",
        }
    ):
        cold = TimeDrivenScene()
        cold.render()
        reference = decoded_frames(cold.manager.file_writer.final_file_path)
        assert cold.time == 2
        assert len(set(cold.renderer.animations_hashes)) == 4
        assert len(reference) == 8
        assert not np.array_equal(reference[4], reference[6])

        warm = TimeDrivenScene()
        writer = warm._get_manager().file_writer
        hits = []
        lookup = writer.is_already_cached

        def cached(key):
            result = lookup(key)
            hits.append(result)
            return result

        writer.is_already_cached = cached
        warm.render()
        assert any(hits)
        assert warm.time == cold.time
        actual = decoded_frames(writer.final_file_path)
        np.testing.assert_array_equal(actual, reference)

        with tempconfig({"disable_caching": True}):
            uncached = TimeDrivenScene()
            uncached.render()
        assert uncached.time == cold.time
        np.testing.assert_array_equal(
            decoded_frames(uncached.manager.file_writer.final_file_path), reference
        )


@pytest.mark.parametrize("backend", ["cairo", "opengl"])
def test_rate_change_cannot_publish_mismatched_video(tmp_path, backend):
    with tempconfig(
        {
            "renderer": backend,
            "format": "mp4",
            "media_dir": str(tmp_path),
            "frame_rate": 4,
            "pixel_width": 64,
            "pixel_height": 32,
            "live_preview": False,
        }
    ):
        scene = TimeDrivenScene()
        manager = scene._get_manager()
        try:
            with (
                tempconfig({"frame_rate": 8}),
                pytest.raises(ValueError, match="frame_rate changed"),
            ):
                scene.render()
            assert manager._file_writer is None
            assert not list(tmp_path.rglob("*.mp4"))
        finally:
            manager.close()


@pytest.mark.parametrize("backend", ["cairo", "opengl"])
def test_stopped_events_do_not_reuse_an_unknown_cached_span(backend, monkeypatch):
    with tempconfig(
        {
            "renderer": backend,
            "format": "none",
            "frame_rate": 4,
            "pixel_width": 32,
            "pixel_height": 16,
            "live_preview": False,
            "disable_caching": False,
            "progress_bar": "none",
        }
    ):
        scene = Scene()
        with scene._get_manager() as manager:
            lookup = Mock(
                side_effect=AssertionError("stop span is not in cache metadata")
            )
            monkeypatch.setattr(manager.file_writer, "is_already_cached", lookup)
            scene.wait(1, stop_condition=lambda: scene.time >= 0.5)
            assert scene.time == 0.5
            lookup.assert_not_called()
