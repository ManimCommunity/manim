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
def test_excluded_plays_leave_later_plays_on_a_full_render_timeline(tmp_path, backend):
    """Frames rendered under -n must equal the same frames of a full render.

    TimeDriven positions its square from scene.time, so a clock that drifted over the
    excluded plays would move every later frame.
    """
    with tempconfig(
        {
            "renderer": backend,
            "format": "mp4",
            "media_dir": str(tmp_path),
            "frame_rate": 4,
            "pixel_width": 64,
            "pixel_height": 32,
            "live_preview": False,
            "disable_caching": True,
            "progress_bar": "none",
        }
    ):
        full = TimeDrivenScene()
        full.render()
        reference = decoded_frames(full.manager.file_writer.final_file_path)
        assert len(reference) == 8

        with tempconfig({"from_animation_number": 2, "upto_animation_number": 3}):
            partial = TimeDrivenScene()
            partial.render()
            actual = decoded_frames(partial.manager.file_writer.final_file_path)

        # Plays 0 and 1 are excluded; plays 2 and 3 supply reference frames 4 through 7.
        assert partial.time == full.time
        np.testing.assert_array_equal(actual, reference[4:])


@pytest.mark.parametrize("backend", ["cairo", "opengl"])
def test_segments_rendered_under_exclusion_are_reused_by_a_full_render(
    tmp_path, backend
):
    """The point of the whole-frame clock: -n output is cache-compatible."""
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
        with tempconfig({"from_animation_number": 2, "upto_animation_number": 3}):
            TimeDrivenScene().render()

        full = TimeDrivenScene()
        writer = full._get_manager().file_writer
        hits = {}
        lookup = writer.is_already_cached

        def cached(key):
            result = lookup(key)
            hits[len(hits)] = result
            return result

        writer.is_already_cached = cached
        full.render()
        # Plays 2 and 3 were rendered by the excluded run and must now be reused.
        assert hits[2]
        assert hits[3]


@pytest.mark.parametrize("backend", ["cairo", "opengl"])
def test_still_output_ends_on_the_full_render_clock(tmp_path, backend):
    """-s excludes every play, so its clock is pure shortcut arithmetic."""
    with tempconfig(
        {
            "renderer": backend,
            "media_dir": str(tmp_path),
            "frame_rate": 4,
            "pixel_width": 64,
            "pixel_height": 32,
            "live_preview": False,
            "disable_caching": True,
            "progress_bar": "none",
        }
    ):
        with tempconfig({"format": "mp4"}):
            full = TimeDrivenScene()
            full.render()
        with tempconfig({"save_last_frame": True}):
            still = TimeDrivenScene()
            still.render()
            assert still.manager.file_writer.output_spec.is_still

    # Every play is excluded here, so this is the pure shortcut clock: four run_time=0.3
    # plays at 4 fps consume two frames each. Before the fix it accumulated 4 * 0.3.
    assert still.time == full.time == 2


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
