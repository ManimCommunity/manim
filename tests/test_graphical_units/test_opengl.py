from __future__ import annotations

from manim import *
from manim.renderer.opengl_renderer import OpenGLRenderer
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
    [i.fix_orientation() for i in (circ, square)]
    triangle.fix_in_frame()


@frames_comparison(
    renderer_class=OpenGLRenderer,
    renderer="opengl",
)
def test_ArrowTip(scene):
    arrows = VGroup()
    for tip_shape in [
        ArrowCircleFilledTip,
        ArrowCircleTip,
        ArrowSquareTip,
        ArrowSquareFilledTip,
        ArrowTriangleTip,
        ArrowTriangleFilledTip,
        StealthTip,
    ]:
        arrows.add(Arrow(tip_shape=tip_shape))
        arrows.add(DoubleArrow(tip_shape_start=tip_shape, tip_shape_end=tip_shape))
    scene.add(arrows.arrange_in_grid(7, 2))
