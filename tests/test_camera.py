from __future__ import annotations

import numpy as np
import pytest

from manim import (
    LEFT,
    ORIGIN,
    RESAMPLING_ALGORITHMS,
    RIGHT,
    UP,
    Camera,
    CapStyleType,
    ImageMobject,
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


@pytest.mark.parametrize(
    ("background_opacity", "expected_center"),
    [(0, [255, 0, 0, 128]), (1, [128, 0, 127, 255])],
)
def test_image_mobject_preserves_background(background_opacity, expected_center):
    camera = Camera(
        pixel_width=80,
        pixel_height=48,
        frame_width=8,
        background_color="#0000FF",
        background_opacity=background_opacity,
    )
    background = camera.pixel_array.copy()
    pixels = np.full((2, 2, 4), [255, 0, 0, 128], dtype=np.uint8)
    image = ImageMobject(pixels)
    image.height = 2
    # Avoid alpha premultiplication rounding from interpolating resamplers.
    image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    camera.capture_mobjects([image])

    assert camera.pixel_array.shape == (48, 80, 4)
    np.testing.assert_array_equal(camera.pixel_array[0], background[0])
    np.testing.assert_array_equal(camera.pixel_array[-1], background[-1])
    np.testing.assert_array_equal(camera.pixel_array[24, 40], expected_center)


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
