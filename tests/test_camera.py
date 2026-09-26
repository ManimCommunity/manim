from __future__ import annotations

import numpy as np
import pytest

from manim import (
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    Camera,
    CapStyleType,
    LineJointType,
    MovingCamera,
    Square,
    VMobject,
)


def test_movingcamera_auto_zoom():
    camera = MovingCamera()
    square = Square()
    margin = 0.5
    camera.auto_zoom([square], margin=margin, animate=False)
    assert camera.frame.height == square.height + margin


def _angled_line(**kwargs):
    return VMobject(stroke_width=20, **kwargs).set_points_as_corners(
        [LEFT, ORIGIN, LEFT + UP]
    )


@pytest.mark.parametrize(
    "styled_kwargs",
    [
        {"cap_style": CapStyleType.ROUND},
        {"cap_style": CapStyleType.SQUARE},
        {"joint_type": LineJointType.ROUND},
        {"joint_type": LineJointType.BEVEL},
    ],
)
def test_auto_stroke_style_does_not_inherit_previous_style(styled_kwargs):
    # The two mobjects do not overlap, so the drawing order must not matter.
    styled = _angled_line(**styled_kwargs).shift(3 * LEFT)
    default = _angled_line().shift(3 * RIGHT)

    styled_first = Camera()
    styled_first.capture_mobjects([styled, default])
    default_first = Camera()
    default_first.capture_mobjects([default, styled])

    np.testing.assert_array_equal(styled_first.pixel_array, default_first.pixel_array)
