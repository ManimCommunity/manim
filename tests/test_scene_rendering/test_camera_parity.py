"""Both renderers draw what the same camera sees."""

import numpy as np
import pytest

from manim import (
    BLUE,
    DEGREES,
    LEFT,
    OUT,
    RIGHT,
    Scene,
    Square,
    ThreeDScene,
    tempconfig,
)


def _three_d(scene):
    scene.set_camera_orientation(
        phi=65 * DEGREES, theta=-50 * DEGREES, zoom=0.9, frame_center=0.5 * RIGHT
    )
    scene.add(Square(3, fill_opacity=1, stroke_width=0, color=BLUE).shift(OUT))
    scene.add(Square(1, fill_opacity=1, stroke_width=0).shift(2 * LEFT))


def _moving_camera(scene):
    scene.add(Square(2, fill_opacity=1, stroke_width=0, color=BLUE))
    scene.add(Square(1, fill_opacity=1, stroke_width=0).shift(3 * RIGHT))
    scene.camera.frame.scale(0.7).shift(LEFT)


def _covered(renderer, scene_class, build):
    with tempconfig(
        {"renderer": renderer, "dry_run": True, "pixel_width": 384, "pixel_height": 216}
    ):
        scene = scene_class()
        try:
            build(scene)
            pixels = np.asarray(scene.get_image().convert("L"))
        finally:
            scene.renderer.close()
    return pixels > 127


@pytest.mark.parametrize(
    ("scene_class", "build"),
    [(ThreeDScene, _three_d), (Scene, _moving_camera)],
    ids=["three_d", "moving_camera"],
)
def test_renderers_place_content_where_the_camera_sees_it(scene_class, build):
    cairo = _bounds(_covered("cairo", scene_class, build))
    opengl = _bounds(_covered("opengl", scene_class, build))

    # Same placement; the current OpenGL fill extends up to 2 pixels further per side.
    np.testing.assert_allclose(opengl.mean(axis=1), cairo.mean(axis=1), atol=0.5)
    np.testing.assert_allclose(np.ptp(opengl, axis=1), np.ptp(cairo, axis=1), atol=4)


def _bounds(mask):
    assert mask.sum() > 2000
    ys, xs = np.nonzero(mask)
    return np.array([[xs.min(), xs.max()], [ys.min(), ys.max()]])


@pytest.mark.parametrize("renderer", ["cairo", "opengl"])
def test_renderers_clear_to_the_camera_background(renderer):
    with tempconfig(
        {
            "renderer": renderer,
            "dry_run": True,
            "pixel_width": 64,
            "pixel_height": 36,
            "background_color": "#000000",
        }
    ):
        scene = Scene()
        try:
            scene.camera.background_color = "#4080c0"
            scene.camera.background_opacity = 0.5
            pixel = np.asarray(scene.get_image())[0, 0]
        finally:
            scene.renderer.close()

    np.testing.assert_allclose(pixel, [0x40, 0x80, 0xC0, 127], atol=1)
