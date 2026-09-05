"""A failing Window constructor must close its acquired native window."""

from unittest.mock import Mock

import pytest

from manim.renderer.opengl import OpenGLRenderer


@pytest.fixture
def window_renderer():
    # Native windows can receive real input during these tests. Supply the event
    # delegation surface without starting an unrelated rendering session.
    return Mock(spec=OpenGLRenderer, pressed_keys=set(), scene=Mock())


@pytest.mark.parametrize("stage", ["context", "position"])
@pytest.mark.parametrize("failure_type", [ValueError, KeyboardInterrupt, SystemExit])
def test_failed_window_constructor_closes_native_window(
    using_temp_opengl_config, monkeypatch, window_renderer, stage, failure_type
):
    from manim.renderer.opengl.window import Window

    failure = failure_type("window initialization failed")
    method = "init_mgl_context" if stage == "context" else "find_initial_position"
    initializing = []

    def fail(window, *args, **kwargs):
        initializing.append(window)
        raise failure

    monkeypatch.setattr(Window, method, fail)
    closed = []
    close = Window.close

    def record_close(window):
        closed.append(window)
        close(window)

    monkeypatch.setattr(Window, "close", record_close)
    try:
        with pytest.raises(failure_type) as caught:
            Window(window_renderer, window_size=(64, 64))
        assert caught.value is failure
        assert len(closed) == 1
        assert closed[0]._window.context is None
    finally:
        for window in initializing:
            if window._window.context is not None:
                close(window)


def test_window_cleanup_error_preserves_constructor_interruption(
    using_temp_opengl_config, monkeypatch, window_renderer
):
    from manim.renderer.opengl.window import Window

    failure = KeyboardInterrupt("window initialization interrupted")
    initializing = []

    def fail(window):
        initializing.append(window)
        raise failure

    monkeypatch.setattr(Window, "init_mgl_context", fail)
    closed = []
    close = Window.close

    def fail_after_close(window):
        close(window)
        closed.append(window)
        raise SystemExit("window cleanup interrupted")

    monkeypatch.setattr(Window, "close", fail_after_close)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            Window(window_renderer, window_size=(64, 64))
        assert caught.value is failure
        assert len(closed) == 1
        assert closed[0]._window.context is None
    finally:
        for window in initializing:
            if window._window.context is not None:
                close(window)
