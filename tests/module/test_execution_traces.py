"""Characterize legacy execution before moving its owner (not a new schedule)."""

import pytest

from manim import Animation, Scene, Square, tempconfig


@pytest.mark.parametrize("backend", ["cairo", "opengl"])
@pytest.mark.parametrize("duration", [1, 0.3])
@pytest.mark.parametrize("skip", [False, True])
def test_legacy_animation_clock_trace(backend, duration, skip):
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

    count = 4 if duration == 1 else 2
    start = duration if skip and backend == "cairo" else 0
    assert trace[:2] == [("begin", start), ("interpolate", start, 0)]
    updates = [item for item in trace if item[0] == "update"]
    if skip:
        assert updates == [("update", start, duration)]
    else:
        expected = [
            ("update", i / 4 if backend == "cairo" else 0, 0 if i == 0 else 0.25)
            for i in range(count)
        ]
        assert updates == expected
    finish_time = (duration if skip else count / 4) if backend == "cairo" else 0
    assert ("finish", finish_time) in trace
    assert ("cleanup", finish_time) in trace
    assert final_time == (count / 4 if backend == "cairo" and not skip else duration)
    assert plays == 1


@pytest.mark.parametrize("backend", ["cairo", "opengl"])
def test_legacy_frozen_wait_clock_trace(backend):
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
        with scene._get_manager():
            scene.wait(0.3, frozen_frame=True)
            assert scene.time == (0.25 if backend == "cairo" else 0.3)
            assert scene.renderer.num_plays == 1


@pytest.mark.parametrize("backend", ["cairo", "opengl"])
def test_legacy_stop_condition_observes_backend_time(backend):
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
            assert scene.time == (0.5 if backend == "cairo" else 1)
    assert stops == ([0.25, 0.5] if backend == "cairo" else [0, 0])
    assert updates == (
        [(0, 0), (0.25, 0.25)] if backend == "cairo" else [(0, 0), (0, 0.25)]
    )
