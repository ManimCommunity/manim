"""Regression coverage for the transparent ImageMobject compositing canvas."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from manim import RESAMPLING_ALGORITHMS, RIGHT, Camera, ImageMobject


@pytest.mark.parametrize("size", [(80, 48), (48, 80)])
@pytest.mark.parametrize("background_alpha", [0, 0.5, 1])
@pytest.mark.parametrize("image_alpha", [0, 128, 255])
@pytest.mark.parametrize("shift", [0, 3.5, -3.5])
def test_image_canvas_preserves_background_and_composites_alpha(
    size, background_alpha, image_alpha, shift
):
    width, height = size
    camera = Camera(
        pixel_width=width,
        pixel_height=height,
        frame_width=8,
        background_color="#204060",
        background_opacity=background_alpha,
    )
    background = camera.pixel_array.copy()
    pixels = np.full((8, 8, 4), [240, 60, 30, image_alpha], dtype=np.uint8)
    image = ImageMobject(pixels).stretch_to_fit_width(2).stretch_to_fit_height(2)
    # Avoid alpha premultiplication rounding from interpolating resamplers.
    image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    image.shift(shift * RIGHT)
    camera.capture_mobjects([image])
    result = camera.pixel_array
    assert result.shape == (height, width, 4)
    assert result.dtype == np.uint8
    # Every edge case leaves the top and bottom outside the image untouched.
    np.testing.assert_array_equal(result[0], background[0])
    np.testing.assert_array_equal(result[-1], background[-1])
    # Sample well inside the visible region, including both clipped placements.
    row = height // 2
    col = int(width / 2 + shift * width / 8)
    expected = Image.alpha_composite(
        Image.fromarray(background[row : row + 1, col : col + 1]),
        Image.fromarray(pixels[:1, :1]),
    )
    np.testing.assert_array_equal(result[row, col], np.asarray(expected)[0, 0])
    if image_alpha == 0:
        np.testing.assert_array_equal(result, background)


@pytest.mark.parametrize("angle", [0, np.pi / 6, np.pi / 2, np.pi])
def test_image_rotation_leaves_transparent_corners(angle):
    camera = Camera(pixel_width=80, pixel_height=48, background_opacity=0)
    image = ImageMobject(np.full((8, 8, 4), [240, 60, 30, 128], dtype=np.uint8))
    image.height = 2
    image.rotate(angle)
    camera.capture_mobjects([image])
    assert camera.pixel_array[24, 40, 3] == 128
    assert camera.pixel_array[0, 0, 3] == 0
    assert camera.pixel_array[-1, -1, 3] == 0
