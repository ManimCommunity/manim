from __future__ import annotations

from manim import *
from manim.constants import DEGREES, LEFT, RIGHT, UR
from manim.mobject.geometry.arc import Circle
from manim.mobject.geometry.polygram import Square, Triangle
from manim.renderer.opengl_renderer import OpenGLRenderer
from manim.scene.scene import Scene
from manim.utils.color import RED
from manim.utils.testing.frames_comparison import frames_comparison

__module_test__ = "opengl"


@frames_comparison(renderer_class=OpenGLRenderer, renderer="opengl")
def test_Circle(scene):
    circle = Circle().set_color(RED)
    scene.add(circle)
    scene.wait()


@frames_comparison(
    renderer_class=OpenGLRenderer,
    renderer="opengl",
)
def test_FixedMobjects3D(scene: Scene):
    scene.renderer.camera.set_euler_angles(phi=75 * DEGREES, theta=-45 * DEGREES)
    circ = Circle(fill_opacity=1).to_edge(LEFT)
    square = Square(fill_opacity=1).to_edge(RIGHT)
    triangle = Triangle(fill_opacity=1).to_corner(UR)
    for i in (circ, square):
        i.fix_orientation()
    triangle.fix_in_frame()
