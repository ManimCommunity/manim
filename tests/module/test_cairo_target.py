from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest
from PIL import Image

from manim import Camera
from manim.renderer.cairo.rendering import _CairoDrawingContext
from manim.renderer.cairo.target import _CairoRasterSettings, _CairoRenderTarget


@pytest.fixture
def target():
    target = _CairoRenderTarget(
        _CairoRasterSettings(
            pixel_width=8,
            pixel_height=4,
            base_pixel_width=8,
            base_pixel_height=4,
        )
    )
    try:
        yield target
    finally:
        target.close()


@pytest.mark.parametrize(
    "operation",
    [
        lambda target: target.pixels,
        lambda target: target.reset(Camera(background_image="missing.png")),
        lambda target: target.clear(),
        lambda target: target.get_scratch_target(),
        lambda target: target.set_pixels(np.zeros((4, 8, 4), dtype=np.uint8)),
        lambda target: target.get_context(Camera()),
        lambda target: target.get_background_image("missing.png"),
        lambda target: target.read_pixels(),
        lambda target: _CairoDrawingContext(
            camera=Camera(), target=target, image_resolver=lambda _: None
        ).draw([]),
    ],
    ids=[
        "pixels",
        "reset",
        "clear",
        "scratch",
        "set",
        "context",
        "background",
        "read",
        "draw",
    ],
)
def test_closed_target_rejects_use(target, operation):
    target.close()
    with pytest.raises(RuntimeError, match="closed"):
        operation(target)


def test_transferred_pixels_survive_close(target):
    target.reset(Camera())
    pixels = target.read_pixels()
    expected = pixels.copy()
    target.close()
    np.testing.assert_array_equal(pixels, expected)


def test_close_releases_owned_buffers_and_scratch(target):
    camera = Camera()
    target.reset(camera)
    target.get_context(camera)
    target.get_background_image(Image.new("RGBA", (8, 4)))
    scratch = target.get_scratch_target()
    scratch.reset(camera)
    scratch.get_context(camera)
    buffers = [
        weakref.ref(array)
        for array in (
            target.pixels,
            target._background,
            scratch.pixels,
            scratch._background,
            *target.background_image_cache.values(),
        )
    ]

    target.close()
    target.close()
    gc.collect()

    assert all(reference() is None for reference in buffers)
    assert target.background_image_cache == {}
    with pytest.raises(RuntimeError, match="closed"):
        _ = scratch.pixels
