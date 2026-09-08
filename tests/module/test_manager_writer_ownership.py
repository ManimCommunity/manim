"""The selected writer is created on first use and reused for the scene."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from manim import CairoRenderer, Manager, Scene, Wait, tempconfig
from manim.renderer.opengl import OpenGLRenderer
from manim.scene.scene_file_writer import SceneFileWriter


@pytest.fixture
def writer_scene(tmp_path, request):
    backend, output = getattr(request, "param", ("cairo", "png"))
    with tempconfig(
        {
            "renderer": backend,
            "format": output,
            "dry_run": False,
            "live_preview": False,
            "log_to_file": False,
            "media_dir": str(tmp_path / "media"),
            "pixel_width": 64,
            "pixel_height": 32,
            "disable_caching": True,
        }
    ):
        factory = Mock(wraps=SceneFileWriter)
        renderer_class = CairoRenderer if backend == "cairo" else OpenGLRenderer
        scene = Scene(renderer=renderer_class(file_writer_class=factory))
        try:
            yield scene, factory
        finally:
            if scene.manager is not None:
                scene.manager.close()
            else:
                scene.renderer.close()


@pytest.mark.parametrize(
    "writer_scene", [("cairo", "png"), ("opengl", "png")], indirect=True
)
def test_snapshot_is_output_free_and_writer_uses_captured_settings(
    writer_scene, tmp_path
):
    scene, factory = writer_scene
    factory.assert_not_called()
    assert scene.get_image().size == (64, 32)
    factory.assert_not_called()
    assert not (tmp_path / "media").exists()

    with tempconfig({"format": "mp4", "media_dir": str(tmp_path / "later")}):
        writer = scene.renderer.file_writer
    assert writer is scene.manager.file_writer is scene.renderer.file_writer
    assert writer.settings == scene.file_writer_settings
    assert writer.output_spec.is_still
    factory.assert_called_once_with(scene.file_writer_settings)


def test_render_creates_writer_before_setup_once(writer_scene, monkeypatch):
    scene, factory = writer_scene
    manager = Manager(scene)
    factory.assert_not_called()

    def setup():
        factory.assert_called_once_with(scene.file_writer_settings)
        assert manager.file_writer is scene.renderer.file_writer

    monkeypatch.setattr(scene, "setup", setup)
    manager.render()
    factory.assert_called_once()
    assert manager.file_writer.final_file_path.exists()


@pytest.mark.parametrize("writer_scene", [("cairo", "none")], indirect=True)
def test_preview_validation_does_not_create_writer_for_cleanup(writer_scene):
    scene, factory = writer_scene
    with pytest.raises(ValueError, match="requires a media artifact"):
        Manager(scene).render(preview=True)
    factory.assert_not_called()


@pytest.mark.parametrize("failure_type", [ValueError, KeyboardInterrupt, SystemExit])
def test_failed_writer_creation_preserves_identity_without_retrying_in_cleanup(
    writer_scene, failure_type
):
    scene, factory = writer_scene
    failure = failure_type("writer initialization failed")
    factory.side_effect = failure
    with pytest.raises(failure_type) as caught:
        Manager(scene).render()
    assert caught.value is failure
    factory.assert_called_once()


def test_recursive_creation_fails_clearly_and_leaves_retry_possible(writer_scene):
    scene, factory = writer_scene
    manager = Manager(scene)
    factory.side_effect = lambda settings: manager.file_writer
    with pytest.raises(RuntimeError, match="Recursive file writer creation"):
        manager.file_writer
    factory.assert_called_once()
    factory.side_effect = None
    assert isinstance(manager.file_writer, SceneFileWriter)
    assert factory.call_count == 2


@pytest.mark.parametrize("writer_scene", [("cairo", "mp4")], indirect=True)
def test_direct_plays_keep_the_same_writer(writer_scene):
    scene, factory = writer_scene
    scene.play(Wait(0.1))
    writer = scene.manager.file_writer
    with pytest.raises(AttributeError):
        scene.renderer.file_writer = Mock()
    scene.play(Wait(0.1))
    writer.join_all_encode_jobs()
    factory.assert_called_once()
    assert writer is scene.renderer.file_writer
    assert len(writer.partial_movie_files) == 2
    assert all(Path(path).is_file() for path in writer.partial_movie_files)
