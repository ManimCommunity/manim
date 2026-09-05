"""Writer ownership/demand, without changing execution-start settings resolution."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from manim import CairoRenderer, Manager, Scene, Wait, tempconfig
from manim.scene.scene_file_writer import SceneFileWriter


@pytest.fixture
def writer_scene(tmp_path, request):
    with tempconfig(
        {
            "format": getattr(request, "param", "png"),
            "dry_run": False,
            "log_to_file": False,
            "media_dir": str(tmp_path / "media"),
            "pixel_width": 64,
            "pixel_height": 32,
            "disable_caching": True,
        }
    ):
        factory = Mock(wraps=SceneFileWriter)
        scene = Scene(renderer=CairoRenderer(file_writer_class=factory))
        try:
            yield scene, factory
        finally:
            # Cleanup must not itself demand a writer.
            writer = getattr(scene.manager, "_file_writer", None)
            if writer is not None:
                writer.abort_encode_jobs()
            scene.renderer.close()


def test_construction_and_snapshot_do_not_create_output(writer_scene, tmp_path):
    scene, factory = writer_scene
    assert scene.manager is None
    factory.assert_not_called()
    assert scene.get_image().size == (64, 32)
    assert scene.manager._file_writer is None
    factory.assert_not_called()
    assert not (tmp_path / "media").exists()


def test_render_creates_writer_before_setup_once(writer_scene, monkeypatch):
    scene, factory = writer_scene
    manager = Manager(scene)
    factory.assert_not_called()

    def setup():
        factory.assert_called_once_with(scene.file_writer_settings)
        assert manager._file_writer is scene.renderer.file_writer

    monkeypatch.setattr(scene, "setup", setup)
    manager.render()
    assert manager.file_writer is scene.renderer.file_writer
    factory.assert_called_once()
    assert manager.file_writer.final_file_path.exists()


def test_explicit_renderer_access_attaches_single_owner(writer_scene):
    scene, factory = writer_scene
    writer = scene.renderer.file_writer
    assert scene.manager.file_writer is writer
    assert scene.renderer.file_writer is writer
    factory.assert_called_once_with(scene.file_writer_settings)
    replacement = Mock()
    scene.renderer.file_writer = replacement
    assert scene.manager.file_writer is replacement
    assert scene.renderer.file_writer is replacement
    factory.assert_called_once()


def test_writer_uses_construction_time_settings_until_resolution_moves(
    writer_scene, tmp_path
):
    scene, factory = writer_scene
    settings = scene.file_writer_settings
    with tempconfig({"format": "mp4", "media_dir": str(tmp_path / "later")}):
        writer = scene.renderer.file_writer
    assert writer.settings is settings
    assert writer.output_spec.is_still
    factory.assert_called_once_with(settings)


def test_owned_writer_survives_backend_rebinding(writer_scene):
    scene, factory = writer_scene
    writer = scene.renderer.file_writer

    class SecondScene(Scene):
        pass

    second = SecondScene(renderer=scene.renderer)
    second_writer = second.renderer.file_writer
    try:
        assert scene.manager.file_writer is writer
        assert second.manager.file_writer is second_writer
        assert second_writer is not writer
        assert writer.output_plan is scene.output_plan
        assert second_writer.output_plan is second.output_plan
        assert factory.call_count == 2
    finally:
        second_writer.abort_encode_jobs()


@pytest.mark.parametrize("writer_scene", ["none"], indirect=True)
def test_preview_validation_does_not_create_writer_for_cleanup(writer_scene):
    scene, factory = writer_scene
    manager = Manager(scene)
    with pytest.raises(ValueError, match="requires a media artifact"):
        manager.render(preview=True)
    assert manager._file_writer is None
    factory.assert_not_called()


@pytest.mark.parametrize("failure_type", [ValueError, KeyboardInterrupt, SystemExit])
def test_failed_writer_creation_preserves_identity_without_retrying_in_cleanup(
    writer_scene, failure_type
):
    scene, factory = writer_scene
    failure = failure_type("writer initialization failed")
    factory.side_effect = failure
    manager = Manager(scene)
    with pytest.raises(failure_type) as caught:
        manager.render()
    assert caught.value is failure
    assert manager._file_writer is None
    factory.assert_called_once()


def test_recursive_creation_fails_clearly_and_leaves_retry_possible(writer_scene):
    scene, factory = writer_scene
    manager = Manager(scene)
    factory.side_effect = lambda settings: manager.file_writer
    with pytest.raises(RuntimeError, match="Recursive file writer creation"):
        manager.file_writer
    assert manager._file_writer is None
    factory.assert_called_once()
    factory.side_effect = None
    assert isinstance(manager.file_writer, SceneFileWriter)
    assert factory.call_count == 2


@pytest.mark.parametrize("writer_scene", ["mp4"], indirect=True)
def test_replacement_retires_previous_unsealed_segment(writer_scene, tmp_path):
    scene, _ = writer_scene
    writer = scene.renderer.file_writer
    path = tmp_path / "replaced.mp4"
    writer.open_partial_movie_stream(animation_index=0, file_path=path)
    job = writer._current_encode_job
    writer.write_frame(np.zeros((32, 64, 4), dtype=np.uint8))
    replacement = Mock()
    try:
        scene.renderer.file_writer = replacement
        assert not job.thread.is_alive()
        assert writer._current_encode_job is None
        assert not path.exists()
        assert scene.manager.file_writer is replacement
    finally:
        writer.abort_encode_jobs()


@pytest.mark.parametrize("writer_scene", ["mp4"], indirect=True)
def test_direct_play_still_creates_legacy_video_segments(writer_scene):
    scene, factory = writer_scene
    scene.play(Wait(0.1))
    writer = scene.manager.file_writer
    writer.join_all_encode_jobs()
    factory.assert_called_once()
    assert writer is scene.renderer.file_writer
    assert len(writer.partial_movie_files) == 1
    assert Path(writer.partial_movie_files[0]).is_file()
