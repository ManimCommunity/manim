"""Cairo and OpenGL expose the same animation times to user code."""

import numpy as np
import pytest

from manim import Animation, Scene, Square, tempconfig


@pytest.mark.parametrize("backend", ["cairo", "opengl"])
@pytest.mark.parametrize("duration", [1, 0.3])
@pytest.mark.parametrize("skip", [False, True])
def test_animation_clock_trace(backend, duration, skip):
    trace = []
    with tempconfig(
        {
            "renderer": backend,
            "format": "none",
            "live_preview": False,
            "pixel_width": 32,
            "pixel_height": 16,
            "frame_rate": 4,
            "disable_caching": True,
            "progress_bar": "none",
        }
    ):
        scene = Scene(skip_animations=skip)
        # OpenGL historically ignores Scene's skip_animations constructor argument.
        scene.renderer._original_skipping_status = skip
        scene.add(Square())

        class Probe(Animation):
            def begin(self):
                trace.append(("begin", scene.time))
                super().begin()

            def interpolate_mobject(self, alpha):
                trace.append(("interpolate", scene.time, float(alpha)))

            def finish(self):
                trace.append(("finish", scene.time))
                super().finish()

            def clean_up_from_scene(self, scene):
                trace.append(("cleanup", scene.time))
                super().clean_up_from_scene(scene)

        scene.add_updater(lambda dt: trace.append(("update", scene.time, float(dt))))
        with scene._get_manager():
            scene.play(Probe(scene.mobjects[0], run_time=duration))
            final_time = scene.time
            plays = scene.renderer.num_plays

    # Skipped and rendered plays consume the same span, so the expectations below no
    # longer branch on `skip`: a shortcut is one interval of that same span.
    count = 4 if duration == 1 else 2
    assert trace[:2] == [("begin", 0), ("interpolate", 0, 0)]
    updates = [item for item in trace if item[0] == "update"]
    if skip:
        # One evaluation step at the start time, with the whole span as its dt.
        assert updates == [("update", 0, duration)]
    else:
        expected = [("update", i / 4, 0 if i == 0 else 0.25) for i in range(count)]
        assert updates == expected
    finish_time = count / 4
    assert ("finish", finish_time) in trace
    assert ("cleanup", finish_time) in trace
    assert final_time == count / 4
    assert plays == 1


@pytest.mark.parametrize("backend", ["cairo", "opengl"])
@pytest.mark.parametrize("skip", [False, True])
def test_frozen_wait_clock_trace(backend, skip):
    with tempconfig(
        {
            "renderer": backend,
            "format": "none",
            "live_preview": False,
            "pixel_width": 32,
            "pixel_height": 16,
            "frame_rate": 4,
            "disable_caching": True,
            "progress_bar": "none",
        }
    ):
        scene = Scene(skip_animations=skip)
        scene.renderer._original_skipping_status = skip
        with scene._get_manager():
            # A skipped frozen wait consumes its whole frames too, not 0.3 s.
            scene.wait(0.3, frozen_frame=True)
            assert scene.time == 0.25
            assert scene.renderer.num_plays == 1


@pytest.mark.parametrize("fps", [4, 15, 24, 30, 60])
@pytest.mark.parametrize(
    "duration", [0.1, 0.3, 0.5, 0.7, 0.9, 1, 1 / 3, 1.1, 1.5, 2, 2.5, 3, 5]
)
def test_sampled_duration_matches_rendered_sample_count(fps, duration):
    """The shortcut advance must equal the span a rendered play actually emits."""
    with tempconfig(
        {
            "format": "none",
            "live_preview": False,
            "pixel_width": 32,
            "pixel_height": 16,
            "frame_rate": fps,
            "progress_bar": "none",
        }
    ):
        scene = Scene()
        with scene._get_manager() as manager:
            rendered_samples = len(np.arange(0, duration, 1 / fps))
            assert manager._sampled_duration(
                duration, frozen=False
            ) * fps == pytest.approx(rendered_samples)


@pytest.mark.parametrize("backend", ["cairo", "opengl"])
def test_stop_condition_observes_consumed_time(backend):
    stops = []
    updates = []
    with tempconfig(
        {
            "renderer": backend,
            "format": "none",
            "live_preview": False,
            "pixel_width": 32,
            "pixel_height": 16,
            "frame_rate": 4,
            "disable_caching": True,
            "progress_bar": "none",
        }
    ):
        scene = Scene()
        scene.add_updater(lambda dt: updates.append((scene.time, float(dt))))

        def stop():
            stops.append(scene.time)
            return len(stops) == 2

        with scene._get_manager():
            scene.wait(1, stop_condition=stop)
            assert scene.time == 0.5
    assert stops == [0.25, 0.5]
    assert updates == [(0, 0), (0.25, 0.25)]
