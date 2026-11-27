from __future__ import annotations

from manim import *
from manim.renderer.opengl import OpenGLRenderer
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
    base_scene=ThreeDScene,
)
def test_FixedMobjects3D(scene: ThreeDScene):
    scene.set_camera_orientation(phi=75 * DEGREES, theta=-135 * DEGREES)
    circ = Circle(fill_opacity=1).to_edge(LEFT)
    square = Square(fill_opacity=1).to_edge(RIGHT)
    triangle = Triangle(fill_opacity=1).to_corner(UR)
    scene.add(Square(2, fill_opacity=0.5, color=BLUE))
    scene.add_fixed_orientation_mobjects(circ, square)
    scene.add_fixed_in_frame_mobjects(triangle)
