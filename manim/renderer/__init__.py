from manim import config
from .cairo_renderer import CairoRenderer
from .opengl_renderer import OpenGLRenderer


def get_default_renderer_class():
    if config["use_opengl_renderer"]:
        return OpenGLRenderer
    return CairoRenderer
