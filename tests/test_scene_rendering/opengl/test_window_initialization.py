"""A failing Window constructor must close its acquired native window."""

import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from manim.renderer.opengl import OpenGLRenderer


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


def test_windows_activation_fails_loudly_without_the_native_context(monkeypatch):
    # The native context is a private Pyglet attribute; if a Pyglet version
    # drops it, activation must say so instead of failing obscurely.
    from manim.renderer.opengl import window as window_module

    bind = Mock()
    context = SimpleNamespace(canvas=SimpleNamespace(hdc=2), set_current=Mock())
    monkeypatch.setattr(window_module.sys, "platform", "win32")
    monkeypatch.setattr(
        window_module.gl, "wgl", SimpleNamespace(wglMakeCurrent=bind), raising=False
    )
    with pytest.raises(RuntimeError, match="native WGL context"):
        window_module._activate_pyglet_context(context)
    bind.assert_not_called()


@pytest.mark.parametrize("operation", ["snapshot", "render"])
def test_first_preview_after_offscreen_retirement(tmp_path, operation):
    # A previous preview can cache WGL function pointers and conceal this failure.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
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
        # Bound hangs while allowing cold imports and driver startup on shared CI.
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def window_renderer():
    # Native windows can receive real input during these tests. Supply the event
    # delegation surface without starting an unrelated rendering session.
    return Mock(spec=OpenGLRenderer, pressed_keys=set(), scene=Mock())


def test_failed_window_constructor_closes_native_window(
    using_temp_opengl_config, monkeypatch, window_renderer
):
    from manim.renderer.opengl.window import Window

    failure = ValueError("window initialization failed")
    method = "init_mgl_context"
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
        with pytest.raises(ValueError) as caught:
            Window(window_renderer, window_size=(64, 64))
        assert caught.value is failure
        assert len(closed) == 1
        assert closed[0]._window.context is None
    finally:
        for window in initializing:
            if window._window.context is not None:
                close(window)


_PRESENT_CHECK = """
import ctypes, sys
from pyglet import gl
from manim import RED, Scene, Square, tempconfig
from manim.renderer.opengl import window as window_module


def native_context():
    # The context the driver draws with, not Pyglet's idea of it.
    if sys.platform == "darwin":
        cgl = ctypes.CDLL("/System/Library/Frameworks/OpenGL.framework/OpenGL")
        cgl.CGLGetCurrentContext.restype = ctypes.c_void_p
        return cgl.CGLGetCurrentContext()
    if sys.platform.startswith("linux"):
        from pyglet.gl import glx

        return ctypes.cast(glx.glXGetCurrentContext(), ctypes.c_void_p).value
    if sys.platform == "win32":
        wgl = ctypes.windll.opengl32
        wgl.wglGetCurrentContext.restype = ctypes.c_void_p
        return wgl.wglGetCurrentContext()
    raise NotImplementedError(sys.platform)


bound = []
swap = window_module.Window.swap_buffers

def checked(self):
    value = ctypes.c_int()
    gl.glGetIntegerv(gl.GL_DRAW_FRAMEBUFFER_BINDING, ctypes.byref(value))
    bound.append((value.value, self.fbo.glo, native_context()))
    swap(self)

window_module.Window.swap_buffers = checked
with tempconfig({"renderer": "opengl", "format": "none", "live_preview": True,
                 "pixel_width": 64, "pixel_height": 36, "window_size": (64, 36)}):
    scene = Scene()
    scene.add(Square(color=RED, fill_opacity=1))
    manager, renderer = scene._get_manager(), scene.renderer
    preview = manager.preview
    preview.show()
    window_context = native_context()
    frame_size = renderer.window.fbo.size
    if sys.argv[1] == "other size":
        frame_size = (frame_size[0] + 2, frame_size[1] + 2)
    from manim.renderer.frame import FrameBuilder
    frame = renderer.render(*FrameBuilder(*frame_size).build(scene.camera, scene.mobjects))
    other = None
    if sys.argv[1] == "after another renderer drew":
        # Another renderer draws offscreen and stays open: its context is current.
        with tempconfig({"live_preview": False}):
            other = Scene()
            other.add(Square())
            other.renderer.render(*other._get_manager()._frame())
        assert native_context() != window_context
    bound.clear()
    preview.present(frame)
    if other is not None:
        other.renderer.close()
    manager.close()
assert bound, bound
assert all(drawn == window for drawn, window, _ in bound), bound
assert all(current == window_context for _, _, current in bound), bound
"""


@pytest.mark.parametrize(
    "case", ["same size", "other size", "after another renderer drew"]
)
def test_present_shows_frames_in_the_window_context(case):
    # Copying an equally sized frame into the window left the frame's framebuffer
    # bound when the window swapped (a black window on macOS), and after an
    # offscreen draw the swap ran with another context current. The window's
    # context and framebuffer must be current at every swap; nothing reads the
    # frame back first.
    result = subprocess.run(
        [sys.executable, "-c", _PRESENT_CHECK, case],
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
