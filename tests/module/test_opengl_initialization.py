"""Acquisition failures must not strand OpenGL contexts or attachments."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from manim.renderer.opengl import OpenGLRenderer
from manim.renderer.opengl import renderer as renderer_module


def initialize(renderer, *, live_preview=False):
    renderer.init_scene(
        Mock(),
        SimpleNamespace(presentation=SimpleNamespace(live_preview=live_preview)),
    )
    renderer.open()


@pytest.fixture
def resources(monkeypatch):
    context = MagicMock()
    color, depth, frame = (Mock() for _ in range(3))
    context.texture.return_value = color
    context.depth_renderbuffer.return_value = depth
    context.framebuffer.return_value = frame
    frame.color_attachments = (color,)
    frame.depth_attachment = depth
    monkeypatch.setattr(
        renderer_module.moderngl, "create_context", Mock(return_value=context)
    )
    return context, color, depth, frame


@pytest.mark.parametrize(
    "stage", ["texture", "depth_renderbuffer", "framebuffer", "use", "enable"]
)
def test_standalone_initialization_rolls_back(resources, stage):
    context, color, depth, frame = resources
    failure = ValueError("initialization failed")
    getattr(frame if stage == "use" else context, stage).side_effect = failure
    renderer = OpenGLRenderer(file_writer_class=Mock())
    with pytest.raises(ValueError) as caught:
        initialize(renderer)
    assert caught.value is failure
    context.release.assert_called_once()
    assert color.release.call_count == (stage != "texture")
    assert depth.release.call_count == (stage not in ["texture", "depth_renderbuffer"])
    assert frame.release.call_count == (stage in ["use", "enable"])
    assert renderer.window is None
    assert renderer._context is None
    assert renderer._frame_buffer_object is None


def test_cleanup_failure_does_not_mask_initialization_or_skip_other_releases(resources):
    context, color, depth, frame = resources
    failure = KeyboardInterrupt("initialization interrupted")
    context.enable.side_effect = failure
    frame.release.side_effect = SystemExit("frame release interrupted")
    depth.release.side_effect = RuntimeError("depth release failed")
    with pytest.raises(KeyboardInterrupt) as caught:
        initialize(OpenGLRenderer(file_writer_class=Mock()))
    assert caught.value is failure
    for resource in resources:
        resource.release.assert_called_once()


def test_attachment_cleanup_failure_preserves_allocation_error(resources):
    context, color, depth, frame = resources
    failure = KeyboardInterrupt("frame allocation interrupted")
    context.framebuffer.side_effect = failure
    depth.release.side_effect = SystemExit("depth release interrupted")
    with pytest.raises(KeyboardInterrupt) as caught:
        initialize(OpenGLRenderer(file_writer_class=Mock()))
    assert caught.value is failure
    for resource in (depth, color, context):
        resource.release.assert_called_once()
    frame.release.assert_not_called()


@pytest.mark.parametrize("failure_type", [KeyboardInterrupt, SystemExit])
def test_context_creation_interrupt_does_not_try_egl(monkeypatch, failure_type):
    failure = failure_type("context creation interrupted")
    factory = Mock(side_effect=failure)
    monkeypatch.setattr(renderer_module.moderngl, "create_context", factory)
    with pytest.raises(failure_type) as caught:
        initialize(OpenGLRenderer(file_writer_class=Mock()))
    assert caught.value is failure
    factory.assert_called_once_with(standalone=True)


@pytest.mark.parametrize("stage", ["detect_framebuffer", "enable"])
def test_window_failure_closes_owner_not_borrowed_context(monkeypatch, stage):
    from manim.renderer.opengl import window as window_module

    window = Mock()
    factory = Mock(return_value=window)
    monkeypatch.setattr(window_module, "Window", factory)
    failure = KeyboardInterrupt("window setup interrupted")
    getattr(window.ctx, stage).side_effect = failure
    renderer = OpenGLRenderer(file_writer_class=Mock())
    with pytest.raises(KeyboardInterrupt) as caught:
        initialize(renderer, live_preview=True)
    assert caught.value is failure
    window.close.assert_called_once()
    window.ctx.release.assert_not_called()
    window.ctx.detect_framebuffer.return_value.release.assert_not_called()
    assert renderer.window is None
