import pytest

from manim.renderer.opengl_renderer import OpenGLRenderer

from .simple_scenes import *


@pytest.mark.parametrize("enable_preview", [False])
def testGetFrameWithPreviewDisabled(use_opengl_renderer):
    """Get frame is able to fetch frame with the correct dimensions when preview is disabled"""
    scene = SquareToCircle()
    assert isinstance(scene.renderer, OpenGLRenderer)
    assert not config.preview

    renderer = scene.renderer
    renderer.update_frame(scene)
    frame = renderer.get_frame()

    assert renderer.get_pixel_shape()[0] == frame.shape[0]
    assert renderer.get_pixel_shape()[1] == frame.shape[1]


@pytest.mark.slow
@pytest.mark.parametrize("enable_preview", [True])
def testGetFrameWithPreviewDisabled(use_opengl_renderer):
    """Get frame is able to fetch frame with the correct dimensions when preview is enabled"""
    scene = SquareToCircle()
    assert isinstance(scene.renderer, OpenGLRenderer)
    assert config.preview is True

    renderer = scene.renderer
    renderer.update_frame(scene)
    frame = renderer.get_frame()

    assert renderer.get_pixel_shape()[0] == frame.shape[0]
    assert renderer.get_pixel_shape()[1] == frame.shape[1]
