"""A failing Window constructor must close its acquired native window."""

import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from manim import config, tempconfig
from manim.renderer.opengl import OpenGLRenderer
from manim.renderer.opengl.window_settings import _WindowSettings


@pytest.mark.parametrize("bound", [True, False])
def test_windows_activation_binds_native_context_before_updating_pyglet(
    monkeypatch, bound
):
    from manim.renderer.opengl import window as window_module

    calls = Mock()
    calls.bind.return_value = bound
    context = SimpleNamespace(
        canvas=SimpleNamespace(hdc=2), _context=1, set_current=calls.set_current
    )
    monkeypatch.setattr(window_module.sys, "platform", "win32")
    monkeypatch.setattr(
        window_module.gl,
        "wgl",
        SimpleNamespace(wglMakeCurrent=calls.bind),
        raising=False,
    )
    if bound:
        window_module._activate_pyglet_context(context)
        assert [call[0] for call in calls.mock_calls] == ["bind", "set_current"]
    else:
        with pytest.raises(RuntimeError, match="Failed to activate"):
            window_module._activate_pyglet_context(context)
        calls.set_current.assert_not_called()
    calls.bind.assert_called_once_with(2, 1)


@pytest.mark.parametrize("operation", ["snapshot", "render"])
def test_first_preview_after_offscreen_retirement(tmp_path, operation):
    # A previous preview can cache WGL function pointers and conceal this failure.
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import faulthandler
import sys
# Diagnose slow imports or native calls before the parent's 20-second deadline.
faulthandler.dump_traceback_later(10, repeat=True)
from manim import Manager, Scene, Square, tempconfig
# Initialize Pyglet's shadow context without opening a Manim preview first.
from manim.renderer.opengl.window import Window
with tempconfig({"renderer": "opengl", "format": "none", "live_preview": False,
                 "pixel_width": 64, "pixel_height": 64, "window_size": (64, 64),
                 "media_dir": sys.argv[2]}):
    scene = Scene()
    scene.add(Square())
    with Manager(scene) as manager:
        if sys.argv[1] == "snapshot":
            assert scene.get_image().size == (64, 64)
        else:
            manager.render()
    with tempconfig({"live_preview": True}):
        preview = Scene()
        preview.add(Square())
        with Manager(preview) as manager:
            manager.render()
            assert preview.renderer.window is not None
            target = preview.renderer.frame_buffer_object
            pixels = target.read(components=4)
            assert any(pixels[::4])  # The white square was actually drawn.
            # Also check restoration of an existing preview after a cold snapshot.
            with tempconfig({"live_preview": False}):
                other = Scene()
                with Manager(other):
                    other.get_image()
            # Do not reactivate the renderer before checking its native framebuffer.
            assert target.read(components=4) == pixels
        assert preview.renderer.window is None
""",
                operation,
                str(tmp_path),
            ],
            capture_output=True,
            encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            timeout=20,
        )
    except subprocess.TimeoutExpired as error:
        # Pytest abbreviates exception locals; emit the child's full stack instead.
        print((error.stderr or b"").decode("utf-8", errors="replace"), file=sys.stderr)
        raise
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def window_renderer():
    # Native windows can receive real input during these tests. Supply the event
    # delegation surface without starting an unrelated rendering session.
    return Mock(spec=OpenGLRenderer, pressed_keys=set(), scene=Mock())


def test_native_window_uses_supplied_settings(
    using_temp_opengl_config, window_renderer
):
    from manim.renderer.opengl.window import Window

    with tempconfig({"window_size": (96, 64)}):
        settings = _WindowSettings.from_config(config)
    with tempconfig({"window_size": (128, 96)}):
        # Compare with the legacy explicit request, not an assumed pixel size:
        # the native backend may scale dimensions on a HiDPI display.
        control = Window(window_renderer, window_size=(96, 64))
        try:
            expected_size = control.size
        finally:
            control.close()
        window = Window(window_renderer, _settings=settings)
        try:
            assert window.size == expected_size
        finally:
            window.close()


@pytest.mark.parametrize(
    ("stage", "failure_type"),
    [("context", ValueError), ("position", ValueError), ("position", SystemExit)],
)
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
