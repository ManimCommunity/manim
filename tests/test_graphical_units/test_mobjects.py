from __future__ import annotations

from manim import *
from manim.utils.testing.frames_comparison import frames_comparison

__module_test__ = "mobjects"


@frames_comparison(base_scene=ThreeDScene)
def test_PointCloudDot(scene):
    p = PointCloudDot()
    scene.add(p)


@frames_comparison
def test_become(scene):
    s = Rectangle(width=2, height=1, color=RED).shift(UP)
    d1, d2, d3 = (Dot() for _ in range(3))

    s1 = s.copy().become(d1, match_width=True).set_opacity(0.25).set_color(BLUE)
    s2 = (
        s.copy()
        .become(d2, match_height=True, match_center=True)
        .set_opacity(0.25)
        .set_color(GREEN)
    )
    s3 = s.copy().become(d3, stretch=True).set_opacity(0.25).set_color(YELLOW)

    scene.add(s, d1, d2, d3, s1, s2, s3)


@frames_comparison
def test_match_style(scene):
    square = Square(fill_color=[RED, GREEN], fill_opacity=1)
    circle = Circle()
    VGroup(square, circle).arrange()
    circle.match_style(square)
    scene.add(square, circle)


@frames_comparison
def test_vmobject_joint_types(scene):
    angled_line = VMobject(stroke_width=20, color=GREEN).set_points_as_corners(
        [
            np.array([-2, 0, 0]),
            np.array([0, 0, 0]),
            np.array([-2, 1, 0]),
        ]
    )
    lines = VGroup(*[angled_line.copy() for _ in range(len(LineJointType))])
    for line, joint_type in zip(lines, LineJointType):
        line.joint_type = joint_type

    lines.arrange(RIGHT, buff=1)
    scene.add(lines)


@frames_comparison
def test_vmobject_cap_styles(scene):
    arcs = VGroup(
        *[
            Arc(
                radius=1,
                start_angle=0,
                angle=TAU / 4,
                stroke_width=20,
                color=GREEN,
                cap_style=cap_style,
            )
            for cap_style in CapStyleType
        ]
    )
    arcs.arrange(RIGHT, buff=1)
    scene.add(arcs)


@frames_comparison
def test_vmobject_radial_gradient(scene):
    circle = Circle(
        radius=2,
        stroke_width=0,
        fill_color=[YELLOW, RED],
        fill_opacity=1,
        fill_gradient_mode="radial",
        radial_gradient_center_inner=ORIGIN,
        radial_gradient_center_outer=RIGHT,
        radial_gradient_inner_radius=0.5,
        radial_gradient_outer_radius=2,
    )
    scene.add(circle)
