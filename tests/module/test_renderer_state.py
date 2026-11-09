"""Every renderer reports where it is in its life through a read-only state."""

from unittest.mock import Mock

import pytest

from manim import Scene, tempconfig
from manim.renderer.cairo import CairoRenderer
from manim.renderer.opengl.renderer import OpenGLRenderer
from manim.renderer.protocol import Renderer, RendererState


@pytest.mark.parametrize("renderer_class", [CairoRenderer, OpenGLRenderer])
def test_renderers_implement_the_protocol(renderer_class):
    renderer = renderer_class()
    try:
        assert isinstance(renderer, Renderer)
        assert renderer.state is RendererState.UNOPENED
        with pytest.raises(AttributeError):
            renderer.state = RendererState.OPEN
    finally:
        renderer.close()
    assert renderer.state is RendererState.CLOSED


SETTINGS = {"pixel_width": 16, "pixel_height": 8, "format": "none"}


@pytest.mark.parametrize("renderer", ["cairo", "opengl"])
def test_a_renderer_goes_from_unopened_to_open_to_closed(renderer):
    with tempconfig({**SETTINGS, "renderer": renderer}):
        renderer = Scene().renderer
        assert renderer.state is RendererState.UNOPENED
        renderer.open()
        assert renderer.state is RendererState.OPEN
        renderer.close()
        assert renderer.state is RendererState.CLOSED
        assert not renderer.state.accepts_work
        with pytest.raises(RuntimeError, match="closed"):
            renderer.open()


def test_a_failed_opengl_close_can_be_retried():
    with tempconfig({**SETTINGS, "renderer": "opengl"}):
        renderer = Scene().renderer
        renderer.open()
        # Releasing fails and leaves the context alive: close() keeps a way to
        # release it and can be called again.
        resources = renderer._resources
        renderer._resources = Mock(close=Mock(side_effect=OSError("release failed")))
        with pytest.raises(OSError, match="release failed"):
            renderer.close()
        resources.pop_all()
        assert renderer.state is RendererState.CLOSE_FAILED
        with pytest.raises(RuntimeError, match="close failed"):
            renderer.open()
        renderer.close()
        assert renderer.state is RendererState.CLOSED
