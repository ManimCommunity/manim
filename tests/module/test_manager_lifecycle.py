"""Lifecycle failures retire output without masking the primary exception."""

from unittest.mock import Mock

import numpy as np
import pytest

from manim import Manager, Scene, tempconfig
from manim.utils.exceptions import EndSceneEarlyException, RerunSceneException


@pytest.mark.parametrize("failure_type", [ValueError, KeyboardInterrupt])
@pytest.mark.parametrize("hook", ["setup", "construct", "tear_down", "post_construct"])
def test_lifecycle_failure_aborts_output_and_preserves_identity(
    dry_run, monkeypatch, hook, failure_type
):
    scene = Scene()
    manager = Manager(scene)
    failure = failure_type("user hook failed")
    abort = Mock()
    monkeypatch.setattr(scene.renderer.file_writer, "abort_encode_jobs", abort)
    monkeypatch.setattr(manager, hook, Mock(side_effect=failure))
    try:
        with pytest.raises(failure_type) as caught:
            manager.render()
        assert caught.value is failure
        abort.assert_called_once()
    finally:
        scene.renderer.close()


@pytest.mark.parametrize(
    ("hook", "failure_type"),
    [("construct", KeyboardInterrupt), ("preview", ValueError)],
)
def test_failure_discards_actual_unsealed_segment(
    tmp_path, monkeypatch, hook, failure_type
):
    with tempconfig(
        {
            "format": "mp4",
            "dry_run": False,
            "media_dir": str(tmp_path / "media"),
            "pixel_width": 64,
            "pixel_height": 32,
            "disable_caching": True,
        }
    ):
        scene = Scene()
        manager = Manager(scene)
        writer = scene.renderer.file_writer
        failure = failure_type("hook interrupted")
        target = tmp_path / "partial.mp4"
        jobs = []

        def fail(*args, **kwargs):
            writer.open_partial_movie_stream(animation_index=0, file_path=target)
            jobs.append(writer._current_encode_job)
            writer.write_frame(np.zeros((32, 64, 4), dtype=np.uint8))
            raise failure

        if hook == "preview":
            # Leave finalization to its own test; inject an outstanding job at
            # the preview boundary to check that it shares the cleanup scope.
            monkeypatch.setattr(manager, "post_construct", lambda: None)
            monkeypatch.setattr("manim.manager.open_media_file", fail)
        else:
            monkeypatch.setattr(manager, hook, fail)
        try:
            with pytest.raises(failure_type) as caught:
                manager.render(preview=hook == "preview")
            assert caught.value is failure
            assert len(jobs) == 1
            assert not jobs[0].thread.is_alive()
            assert writer._current_encode_job is None
            assert not target.exists()
        finally:
            # A regression must fail the test, not leave a non-daemon thread
            # hanging the test runner on exit.
            writer.abort_encode_jobs()
            scene.renderer.close()


@pytest.mark.parametrize("failure_type", [EndSceneEarlyException, RerunSceneException])
@pytest.mark.parametrize("hook", ["setup", "tear_down", "post_construct"])
def test_construction_control_flow_is_not_swallowed_in_other_hooks(
    dry_run, monkeypatch, hook, failure_type
):
    scene = Scene()
    manager = Manager(scene)
    failure = failure_type()
    abort = Mock()
    monkeypatch.setattr(scene.renderer.file_writer, "abort_encode_jobs", abort)
    monkeypatch.setattr(manager, hook, Mock(side_effect=failure))
    try:
        with pytest.raises(failure_type) as caught:
            manager.render()
        assert caught.value is failure
        abort.assert_called_once()
    finally:
        scene.renderer.close()


def test_rerun_reset_failure_still_aborts_output(dry_run, monkeypatch):
    scene = Scene()
    manager = Manager(scene)
    failure = RuntimeError("reset failed")
    abort = Mock()
    monkeypatch.setattr(manager, "construct", Mock(side_effect=RerunSceneException()))
    monkeypatch.setattr(
        scene.renderer, "clear_screen", Mock(side_effect=failure), raising=False
    )
    monkeypatch.setattr(scene.renderer.file_writer, "abort_encode_jobs", abort)
    try:
        with pytest.raises(RuntimeError) as caught:
            manager.render()
        assert caught.value is failure
        abort.assert_called_once()
    finally:
        scene.renderer.close()


def test_cleanup_failure_does_not_replace_primary_exception(dry_run, monkeypatch):
    scene = Scene()
    manager = Manager(scene)
    failure = KeyboardInterrupt("construct interrupted")
    monkeypatch.setattr(manager, "construct", Mock(side_effect=failure))
    abort = Mock(side_effect=RuntimeError("cleanup failed"))
    report = Mock()
    monkeypatch.setattr(scene.renderer.file_writer, "abort_encode_jobs", abort)
    monkeypatch.setattr("manim.manager.logger.exception", report)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            manager.render()
        assert caught.value is failure
        abort.assert_called_once()
        report.assert_called_once()
    finally:
        scene.renderer.close()
