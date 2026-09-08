"""Rendering errors stop encoding jobs and preserve the original exception."""

from unittest.mock import Mock

import numpy as np
import pytest

from manim import Manager, Scene, tempconfig
from manim.utils.exceptions import EndSceneEarlyException, RerunSceneException


@pytest.mark.parametrize(
    ("hook", "failure_type"),
    [
        ("setup", ValueError),
        ("construct", KeyboardInterrupt),
        ("tear_down", ValueError),
        ("post_construct", KeyboardInterrupt),
        ("setup", EndSceneEarlyException),
        ("tear_down", EndSceneEarlyException),
        ("post_construct", EndSceneEarlyException),
        ("setup", RerunSceneException),
        ("tear_down", RerunSceneException),
        ("post_construct", RerunSceneException),
    ],
)
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


def test_failure_discards_actual_unsealed_segment(tmp_path, monkeypatch):
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
        failure = KeyboardInterrupt("construct interrupted")
        target = tmp_path / "partial.mp4"
        jobs = []

        def fail(*args, **kwargs):
            writer.open_partial_movie_stream(animation_index=0, file_path=target)
            jobs.append(writer._current_encode_job)
            writer.write_frame(np.zeros((32, 64, 4), dtype=np.uint8))
            raise failure

        monkeypatch.setattr(manager, "construct", fail)
        try:
            with pytest.raises(KeyboardInterrupt) as caught:
                manager.render()
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


def test_preview_failure_preserves_completed_output(tmp_path, monkeypatch):
    failure = ValueError("media opener failed")
    monkeypatch.setattr("manim.manager.open_media_file", Mock(side_effect=failure))
    with tempconfig(
        {
            "format": "png",
            "media_dir": str(tmp_path),
            "pixel_width": 64,
            "pixel_height": 32,
        }
    ):
        scene = Scene()
        with pytest.raises(ValueError) as caught:
            scene.render(preview=True)
        assert caught.value is failure
        assert scene.manager.file_writer.final_file_path.is_file()
        assert scene.renderer._closed


def test_rerun_cannot_hide_encoder_failure(dry_run, monkeypatch):
    scene = Scene()
    manager = Manager(scene)
    failure = RuntimeError("encoder failed")
    monkeypatch.setattr(manager, "construct", Mock(side_effect=RerunSceneException()))
    monkeypatch.setattr(
        manager.file_writer, "abort_encode_jobs", Mock(side_effect=failure)
    )
    try:
        with pytest.raises(RuntimeError) as caught:
            manager.render()
        assert caught.value is failure
        assert scene.renderer._closed
    finally:
        monkeypatch.undo()
        manager.close()


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
