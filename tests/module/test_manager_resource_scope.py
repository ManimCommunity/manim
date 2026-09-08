"""Manager closes rendering resources and removes the log handler it created."""

import logging
from unittest.mock import Mock

import pytest

from manim import Manager, Scene, logger, tempconfig
from manim._config import logger_utils


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


def test_constructor_and_image_request_do_not_open_log(managed_scene):
    scene = managed_scene
    path = scene._log_file_path
    assert not path.parent.exists()
    scene.get_image()
    assert not path.parent.exists()
    assert scene.manager._log_handler is None


def test_success_retires_without_forcing_readback(managed_scene, monkeypatch):
    scene = managed_scene
    readback = Mock(wraps=scene.renderer.get_frame)
    monkeypatch.setattr(scene.renderer, "get_frame", readback)
    scene.render()
    readback.assert_not_called()
    assert scene.manager._closed
    assert scene.renderer._closed
    with pytest.raises(RuntimeError, match="closed"):
        scene.renderer.get_frame()


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


def test_success_cleanup_failure_is_reported_and_backend_still_retires(
    managed_scene, monkeypatch
):
    scene = managed_scene
    manager = Manager(scene)
    original = manager._close_log_handler
    failure = RuntimeError("log retirement failed")
    calls = []

    def close_log():
        original()
        calls.append(True)
        if len(calls) == 1:
            raise failure

    monkeypatch.setattr(manager, "_close_log_handler", close_log)
    with pytest.raises(RuntimeError) as caught:
        manager.render()
    assert caught.value is failure
    assert scene.renderer._closed
    assert manager._closed


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


@pytest.mark.parametrize("failure_type", [ValueError, KeyboardInterrupt, SystemExit])
def test_render_failure_retires_backend_and_log(
    managed_scene, monkeypatch, failure_type
):
    scene = managed_scene
    failure = failure_type("user failure")
    handler = []

    def construct():
        handler.append(scene.manager._log_handler)
        raise failure

    monkeypatch.setattr(scene, "construct", construct)
    with pytest.raises(failure_type) as caught:
        scene.render()
    assert caught.value is failure
    assert scene.renderer._closed
    assert scene.manager._closed
    assert handler[0] not in logger.handlers
    assert handler[0].stream is None


def test_log_close_failure_does_not_mask_construct_failure(managed_scene, monkeypatch):
    scene = managed_scene
    primary = KeyboardInterrupt("construct interrupted")
    cleanup_error = SystemExit("log cleanup interrupted")

    def setup():
        handler = scene.manager._log_handler
        original = handler.close

        def close():
            original()
            raise cleanup_error

        monkeypatch.setattr(handler, "close", close)

    monkeypatch.setattr(scene, "setup", setup)
    monkeypatch.setattr(scene, "construct", Mock(side_effect=primary))
    with pytest.raises(KeyboardInterrupt) as caught:
        scene.render()
    assert caught.value is primary
    assert scene.renderer._closed
    assert scene.manager._log_handler not in logger.handlers
    monkeypatch.undo()
    scene.manager.close()


def test_body_exception_stays_primary_on_context_exit(dry_run, monkeypatch):
    scene = Scene()
    manager = Manager(scene)
    failure = KeyboardInterrupt("body interrupted")
    abort = Mock(side_effect=RuntimeError("abort failed"))
    monkeypatch.setattr(manager.file_writer, "abort_encode_jobs", abort)
    with pytest.raises(KeyboardInterrupt) as caught, manager:
        raise failure
    assert caught.value is failure
    assert scene.renderer._closed
    abort.assert_called_once()
    monkeypatch.undo()
    manager.close()


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


def test_file_handler_startup_failure_closes_new_handler(tmp_path, monkeypatch):
    handlers = []
    original = logging.FileHandler

    def create(*args, **kwargs):
        handler = original(*args, **kwargs)
        handlers.append(handler)
        return handler

    failure = KeyboardInterrupt("formatter failed")
    monkeypatch.setattr(logging, "FileHandler", create)
    monkeypatch.setattr(logger_utils, "JSONFormatter", Mock(side_effect=failure))
    with pytest.raises(KeyboardInterrupt) as caught:
        logger_utils.set_file_logger(tmp_path / "failed.log")
    assert caught.value is failure
    assert handlers[0].stream is None
    assert handlers[0] not in logger.handlers
