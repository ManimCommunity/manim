"""Cairo raster demand is independent of renderer/Scene construction."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from manim import RED, CairoRenderer, Scene, Square, tempconfig
from manim.renderer.cairo import renderer as cairo_module


@pytest.fixture
def target_factory(monkeypatch):
    factory = Mock(wraps=cairo_module._CairoRenderTarget)
    monkeypatch.setattr(cairo_module, "_CairoRenderTarget", factory)
    return factory


def test_scene_init_and_snapshot_do_not_open_primary_target(dry_run, target_factory):
    with tempconfig({"pixel_width": 64, "pixel_height": 32}):
        scene = Scene()
        try:
            target_factory.assert_not_called()
            scene.add(Square())
            image = scene.get_image()
            assert image.size == (64, 32)
            assert scene.renderer._target is None
            assert target_factory.call_count == 1
            scene.renderer.update_frame(scene)
            assert target_factory.call_count == 2
            np.testing.assert_array_equal(image, scene.renderer.get_frame())
            scene.renderer.update_frame(scene)
            assert target_factory.call_count == 2
        finally:
            scene.renderer.close()


def test_skipped_drawing_and_close_do_not_allocate(target_factory):
    renderer = CairoRenderer(skip_animations=True)
    try:
        renderer.update_frame(None, ignore_skipping=False)
        renderer.save_static_frame_data(SimpleNamespace(moving_mobjects=[]), [])
    finally:
        renderer.close()
    renderer.close()
    with pytest.raises(RuntimeError, match="closed"):
        renderer.get_frame()
    with pytest.raises(RuntimeError, match="closed"):
        renderer.render_mobjects([])
    with pytest.raises(RuntimeError, match="closed"):
        renderer.add_frame(np.zeros((1, 1, 4), dtype=np.uint8))
    target_factory.assert_not_called()


def test_first_readback_allocates_once_and_preserves_initial_buffer(target_factory):
    with tempconfig({"pixel_width": 64, "pixel_height": 32}):
        renderer = CairoRenderer()
    try:
        target_factory.assert_not_called()
        frame = renderer.get_frame()
        np.testing.assert_array_equal(frame, np.zeros((32, 64, 4), dtype=np.uint8))
        frame[:] = 255
        np.testing.assert_array_equal(renderer.get_frame(), 0)
        target_factory.assert_called_once()
    finally:
        renderer.close()


def test_first_draw_uses_captured_settings_not_later_config(target_factory):
    with tempconfig(
        {
            "pixel_width": 64,
            "pixel_height": 32,
            "background_color": RED,
        }
    ):
        renderer = CairoRenderer()
    try:
        with tempconfig({"pixel_width": 128, "pixel_height": 96}):
            renderer.render_mobjects([])
        frame = renderer.get_frame()
        assert frame.shape == (32, 64, 4)
        assert np.all(frame == frame[0, 0])
        np.testing.assert_array_equal(frame[0, 0], RED.to_int_rgba())
        target_factory.assert_called_once_with(renderer._raster_settings)
    finally:
        renderer.close()


@pytest.mark.parametrize("close_after_failure", [False, True])
def test_failed_target_allocation_is_not_retained(monkeypatch, close_after_failure):
    target_type = cairo_module._CairoRenderTarget
    calls = []

    def allocate(settings):
        calls.append(settings)
        if len(calls) == 1:
            raise MemoryError("allocation failed")
        return target_type(settings)

    monkeypatch.setattr(cairo_module, "_CairoRenderTarget", allocate)
    with tempconfig({"pixel_width": 64, "pixel_height": 32}):
        renderer = CairoRenderer()
    try:
        with pytest.raises(MemoryError, match="allocation failed"):
            renderer.get_frame()
        assert renderer._target is None
        if close_after_failure:
            renderer.close()
            with pytest.raises(RuntimeError, match="closed"):
                renderer.get_frame()
            assert len(calls) == 1
        else:
            assert renderer.get_frame().shape == (32, 64, 4)
            assert len(calls) == 2
    finally:
        renderer.close()
