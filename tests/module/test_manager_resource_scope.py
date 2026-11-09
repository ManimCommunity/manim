"""Manager closes rendering resources and removes the log handler it created."""

import logging
from unittest.mock import Mock

import pytest

from manim import Manager, Scene, logger, tempconfig
from manim.manager import _State
from manim.renderer.protocol import RendererState


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
        assert scene.renderer.state is not RendererState.CLOSED
        assert scene.renderer.get_frame().shape == (32, 64, 4)
    assert scene.renderer.state is RendererState.CLOSED
    assert manager._scope_depth == 0


def test_a_failure_in_a_nested_scope_leaves_the_manager_open(managed_scene):
    scene = managed_scene
    manager = Manager(scene)
    with manager:
        with pytest.raises(ValueError), manager:
            raise ValueError("handled by the enclosing block")
        assert scene.renderer.get_frame().shape == (32, 64, 4)
    assert scene.renderer.state is RendererState.CLOSED


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
            assert scene.renderer.state is not RendererState.CLOSED
        assert scene.renderer.state is RendererState.CLOSED
        assert manager._state is _State.CLOSED
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
    assert scene.renderer.state is RendererState.CLOSED
    assert scene.manager._state is _State.CLOSED
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
    assert renderer.state is RendererState.CLOSED


def test_play_after_render_in_scope_keeps_the_render_log(managed_scene, monkeypatch):
    scene = managed_scene
    monkeypatch.setattr(scene, "setup", lambda: logger.info("inside managed setup"))
    with Manager(scene) as manager:
        manager.render()
        scene.wait(0.1)
    assert "inside managed setup" in scene._log_file_path.read_text()


def test_a_manager_renders_its_scene_only_once(managed_scene):
    with Manager(managed_scene) as manager:
        manager.render()
        with pytest.raises(RuntimeError, match="only once"):
            manager.render()


def _logging_config(tmp_path):
    return tempconfig(
        {
            "format": "none",
            "pixel_width": 64,
            "pixel_height": 32,
            "log_to_file": True,
            "log_dir": str(tmp_path / "logs"),
            "media_dir": str(tmp_path / "media"),
        }
    )


def test_the_log_file_keeps_messages_from_the_scene_constructor(tmp_path):
    class LogsWhenCreated(Scene):
        def __init__(self):
            super().__init__()
            logger.warning("logged by the constructor")

    with _logging_config(tmp_path):
        scene = LogsWhenCreated()
        scene.render()
    assert "logged by the constructor" in scene._log_file_path.read_text()


def test_the_log_file_keeps_cleanup_failures_after_a_failed_render(
    tmp_path, monkeypatch
):
    class Fails(Scene):
        def construct(self):
            raise RuntimeError("render failed")

    with _logging_config(tmp_path):
        scene = Fails()
        monkeypatch.setattr(
            scene.renderer, "close", Mock(side_effect=OSError("cannot close"))
        )
        with pytest.raises(RuntimeError, match="render failed"):
            scene.render()
    assert "Failed to clean up" in scene._log_file_path.read_text()


def test_a_failed_close_after_a_render_is_not_cleaned_up_again(
    managed_scene, monkeypatch
):
    # The close at the end of a successful render used to sit inside the
    # render's error handling, which then closed everything a second time.
    close = Mock(side_effect=OSError("close failed"))
    monkeypatch.setattr(managed_scene.renderer, "close", close)
    manager = Manager(managed_scene)
    with pytest.raises(OSError, match="close failed"):
        manager.render()
    assert close.call_count == 1
    assert manager._state is _State.CLOSE_FAILED
    with pytest.raises(RuntimeError, match="close failed"):
        manager.play()
    monkeypatch.undo()
    manager.close()
    assert manager._state is _State.CLOSED


def test_a_failure_to_close_the_log_after_a_failed_render_is_logged(
    managed_scene, monkeypatch
):
    records = []
    recorder = logging.Handler()
    recorder.emit = records.append
    logger.addHandler(recorder)
    monkeypatch.setattr(managed_scene, "construct", Mock(side_effect=ValueError))
    manager = Manager(managed_scene)
    monkeypatch.setattr(
        manager, "_close_log_handler", Mock(side_effect=OSError("disk full"))
    )
    try:
        with pytest.raises(ValueError):
            manager.render()
    finally:
        logger.removeHandler(recorder)
        monkeypatch.undo()
    assert [str(r.exc_info[1]) for r in records if r.exc_info] == ["disk full"]
    assert manager._state is _State.CLOSE_FAILED
    manager.close()
    assert manager._state is _State.CLOSED


def test_a_play_outside_a_render_does_not_leave_the_log_file_open(managed_scene):
    # Nothing would close it: render() and ``with manager:`` blocks do, but a
    # bare play is neither, and the handler would get every later scene's log.
    def file_handlers():
        return [h for h in logger.handlers if isinstance(h, logging.FileHandler)]

    before = file_handlers()
    managed_scene.wait(0.1)
    managed_scene.wait(0.1)
    assert file_handlers() == before
    # Each play appends to the scene's log file.
    log = managed_scene._log_file_path.read_text()
    assert log.count("Log file will be saved") == 2
