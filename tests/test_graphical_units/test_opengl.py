import pytest

from manim import *
from manim.renderer.opengl_renderer import OpenGLRenderer
from tests.test_graphical_units.testing.frames_comparison import frames_comparison

__module_test__ = "opengl"


@frames_comparison(renderer_class=OpenGLRenderer, renderer="opengl")
def test_Circle(scene):
    circle = Circle().set_color(RED)
    scene.add(circle)
    scene.wait()
