import numpy as np

from manim.mobject.opengl.opengl_surface import OpenGLSurface
from manim.mobject.opengl.opengl_three_dimensions import OpenGLSurfaceMesh


def test_surface_initialization(using_opengl_renderer):
    surface = OpenGLSurface(
        lambda u, v: (u, v, u * np.sin(v) + v * np.cos(u)),
        u_range=(-3, 3),
        v_range=(-3, 3),
    )

    mesh = OpenGLSurfaceMesh(surface)
