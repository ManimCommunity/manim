"""Manager closes rendering resources and removes the log handler it created."""

import logging
from unittest.mock import Mock

import pytest

from manim import Manager, Scene, logger, tempconfig


@pytest.fixture(params=["cairo", "opengl"])
def managed_scene(request, tmp_path):
    with tempconfig(
        {
            "renderer": request.param,
            "format": "none",
            "live_preview": False,
            "pixel_width": 64,
            "pixel_height": 32,
            "log_to_file": True,
            "log_dir": str(tmp_path / "logs"),
            "media_dir": str(tmp_path / "media"),
        }
    ):
        scene = Scene()
        try:
            yield scene
        finally:
            if scene.manager is not None:
                scene.manager.close()
            else:
                scene.renderer.close()


def test_nested_inspection_scopes_retire_only_at_outer_exit(managed_scene):
    scene = managed_scene
    manager = Manager(scene)
    with manager:
        with manager:
            scene.render()
        assert not scene.renderer._closed
        assert scene.renderer.get_frame().shape == (32, 64, 4)
    assert scene.renderer._closed
    assert manager._scope_depth == 0


def test_render_log_is_scoped_and_context_exit_retires_backend(
    managed_scene, monkeypatch
):
    scene = managed_scene
    external = logging.NullHandler()
    logger.addHandler(external)
    opened = []

    def setup():
        opened.append(scene.manager._log_handler)
        assert opened[-1] in logger.handlers
        logger.info("inside managed setup")

    monkeypatch.setattr(scene, "setup", setup)
    try:
        with Manager(scene) as manager:
            manager.render()
            assert manager._log_handler is None
            assert opened[0].stream is None
            assert external in logger.handlers
            assert not scene.renderer._closed
        assert scene.renderer._closed
        assert manager._closed
        assert external in logger.handlers
        assert "inside managed setup" in scene._log_file_path.read_text()
    finally:
        logger.removeHandler(external)
        external.close()


def test_render_failure_retires_backend_and_log(managed_scene, monkeypatch):
    scene = managed_scene
    failure = KeyboardInterrupt("user failure")
    handler = []

    def construct():
        handler.append(scene.manager._log_handler)
        raise failure

    monkeypatch.setattr(scene, "construct", construct)
    with pytest.raises(KeyboardInterrupt) as caught:
        scene.render()
    assert caught.value is failure
    assert scene.renderer._closed
    assert scene.manager._closed
    assert handler[0] not in logger.handlers
    assert handler[0].stream is None


def test_scene_renderer_binding_cannot_be_replaced(managed_scene):
    scene = managed_scene
    renderer = scene.renderer
    with scene._get_manager() as manager:
        renderer.update_frame(scene)
        with pytest.raises(RuntimeError, match="already bound"):
            Scene(renderer=renderer)
        with pytest.raises(AttributeError):
            scene.renderer = Mock()
        scene.render()
        assert manager.renderer is renderer
        assert renderer.get_frame().shape == (32, 64, 4)
    assert renderer._closed
