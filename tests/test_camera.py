from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from manim import (
    BLUE,
    RED,
    Camera,
    Group,
    ImageMobjectFromCamera,
    Mobject,
    MovingCamera,
    MultiCamera,
    Scene,
    Square,
    ThreeDCamera,
    config,
    tempconfig,
)
from manim.renderer.cairo import CairoRenderer
from manim.renderer.cairo.rendering import _CairoDrawingContext
from manim.utils.color import color_to_int_rgba


def test_movingcamera_auto_zoom():
    camera = MovingCamera()
    square = Square()
    margin = 0.5
    camera.auto_zoom([square], margin=margin, animate=False)
    assert camera.frame.height == square.height + margin


def test_default_camera_is_movable_and_resource_free():
    camera = Camera()

    assert not hasattr(camera, "pixel_array")
    assert not hasattr(camera, "capture_mobjects")
    camera.frame.move_to([2, 1, 0]).set(width=6)

    assert camera.frame_center.tolist() == [2, 1, 0]
    assert camera.frame_width == 6


def test_camera_frame_geometry_is_semantic_and_explicit():
    camera = Camera(frame_width=8, frame_height=4)
    assert camera.frame_width == 8
    assert camera.frame_height == 4

    custom_frame = Square(side_length=3).move_to([2, 1, 0])
    custom_camera = Camera(frame=custom_frame)
    assert custom_camera.frame is custom_frame
    assert custom_camera.frame_width == 3
    assert custom_camera.frame_height == 3
    np.testing.assert_array_equal(custom_camera.frame_center, [2, 1, 0])


@pytest.mark.parametrize(
    "camera_class", [Camera, MovingCamera, MultiCamera, ThreeDCamera]
)
@pytest.mark.parametrize("pixel_shape", [(128, 128), (96, 160), (192, 96)])
def test_default_camera_preserves_square_geometry(camera_class, pixel_shape):
    width, height = pixel_shape
    with tempconfig({"pixel_width": width, "pixel_height": height}):
        camera = camera_class()
        assert camera.frame_width == pytest.approx(config.frame_width)
        assert camera.frame_width / camera.frame_height == pytest.approx(width / height)
        frame_points = camera.frame.points.copy()
        renderer = CairoRenderer(camera=camera)
        try:
            renderer.render_mobjects(
                [Square(fill_color="#ffffff", fill_opacity=1, stroke_width=0)],
            )
            pixels = renderer.get_frame()
            rows, columns = np.where(pixels[:, :, 0] > 128)
            assert len(rows) > 0
            assert np.ptp(columns) == pytest.approx(np.ptp(rows), abs=1)
            np.testing.assert_array_equal(camera.frame.points, frame_points)
        finally:
            renderer.close()


@pytest.mark.parametrize(
    ("dimensions", "expected_width", "expected_height"),
    [
        ({"frame_width": 6}, 6, 12),
        ({"frame_height": 6}, 3, 6),
        ({"frame_width": 6, "frame_height": 3}, 6, 3),
    ],
)
def test_camera_resolves_only_unspecified_dimensions(
    dimensions, expected_width, expected_height
):
    with tempconfig({"pixel_width": 60, "pixel_height": 120}):
        camera = Camera(**dimensions)
        assert camera.frame_width == pytest.approx(expected_width)
        assert camera.frame_height == pytest.approx(expected_height)


@pytest.mark.parametrize(
    "removed_setting",
    [
        {"pixel_width": 100},
        {"frame_rate": 30},
        {"cairo_line_width_multiple": 0.02},
        {"fixed_dimension": 1},
    ],
)
def test_camera_rejects_removed_raster_settings(removed_setting):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        Camera(**removed_setting)


def test_renderer_rejects_unknown_constructor_settings():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        CairoRenderer(pixel_width=100)


def test_default_scene_camera_auto_zoom():
    with tempconfig({"dry_run": True, "quality": "low_quality"}):
        scene = Scene()
        square = Square().move_to([2, 0, 0])
        scene.play(scene.camera.auto_zoom([square], margin=0.5))

    assert scene.camera.frame_center.tolist() == square.get_center().tolist()
    assert scene.camera.frame_height == square.height + 0.5


def test_mobject_get_image_uses_temporary_renderer():
    with tempconfig({"pixel_width": 32, "pixel_height": 18}):
        image = Square(color=BLUE, fill_opacity=1).get_image()

    pixels = np.asarray(image)
    assert image.size == (32, 18)
    assert np.any(pixels[:, :, 2] > 0)


def test_camera_backed_image_constructs_without_camera_pixels():
    camera = Camera()

    image = ImageMobjectFromCamera(camera)

    assert image.camera is camera
    assert not hasattr(image, "pixel_array")
    assert "get_pixel_array" not in type(image).__dict__
    assert image.width / image.height == pytest.approx(
        camera.frame_width / camera.frame_height,
    )


def test_renderer_owns_background_readback():
    with tempconfig({"pixel_width": 8, "pixel_height": 4}):
        renderer = CairoRenderer(
            camera=Camera(background_color=RED, background_opacity=0.5),
        )
        try:
            renderer.update_frame(None, mobjects=[Mobject()])
            pixels = renderer.get_frame()
            expected = color_to_int_rgba(RED, 0.5)
            assert np.all(pixels == expected)

            pixels[:] = 0
            assert np.all(renderer.get_frame() == expected)
        finally:
            renderer.close()


def test_background_image_is_loaded_by_renderer(tmp_path):
    source = np.array(
        [
            [[255, 0, 0, 255], [0, 255, 0, 192]],
            [[0, 0, 255, 128], [255, 255, 0, 64]],
        ],
        dtype=np.uint8,
    )
    image_path = tmp_path / "asymmetric.png"
    Image.fromarray(source, mode="RGBA").save(image_path)

    with tempconfig({"pixel_width": 2, "pixel_height": 2}):
        camera = Camera(background_image=str(image_path))
        assert not hasattr(camera, "background")
        renderer = CairoRenderer(camera=camera)
        try:
            renderer.update_frame(None, mobjects=[Mobject()])
            np.testing.assert_array_equal(renderer.get_frame(), source)
        finally:
            renderer.close()


def test_background_image_is_resized_to_target(tmp_path):
    source = np.zeros((2, 4, 4), dtype=np.uint8)
    source[:, :2] = [255, 0, 0, 255]
    source[:, 2:] = [0, 0, 255, 255]
    image_path = tmp_path / "wide.png"
    Image.fromarray(source, mode="RGBA").save(image_path)

    with tempconfig({"pixel_width": 2, "pixel_height": 2}):
        renderer = CairoRenderer(camera=Camera(background_image=str(image_path)))
        try:
            renderer.update_frame(None, mobjects=[Mobject()])
            expected = np.asarray(
                Image.fromarray(source, mode="RGBA").resize((2, 2)),
                dtype=np.uint8,
            )
            np.testing.assert_array_equal(renderer.get_frame(), expected)
        finally:
            renderer.close()


def test_background_colored_mobjects_reuse_renderer_scratch_target(tmp_path):
    image_path = tmp_path / "background.png"
    Image.new("RGBA", (8, 4), (20, 40, 60, 255)).save(image_path)

    with tempconfig({"pixel_width": 8, "pixel_height": 4}):
        square = Square().color_using_background_image(str(image_path))
        renderer = CairoRenderer()
        try:
            renderer.update_frame(None, mobjects=[square])
            scratch = renderer._target.get_scratch_target()
            renderer.update_frame(None, mobjects=[square])
            assert renderer._target.get_scratch_target() is scratch
        finally:
            renderer.close()

    with pytest.raises(RuntimeError, match="closed"):
        _ = scratch.pixels


def test_nested_view_excludes_its_own_display():
    nested_camera = Camera()
    view = ImageMobjectFromCamera(nested_camera)
    primary_camera = MultiCamera([view])
    renderer = CairoRenderer(camera=primary_camera)

    try:
        with patch.object(_CairoDrawingContext, "draw", autospec=True) as draw:
            renderer.update_frame(None, mobjects=[Group(view)])
        nested_draw = next(
            call for call in draw.call_args_list if call.args[0].camera is nested_camera
        )
        excluded = nested_draw.kwargs["excluded_mobjects"]
        assert view in excluded
    finally:
        renderer.close()


def test_nested_target_size_tracks_display_size_and_closes():
    with tempconfig({"pixel_width": 100, "pixel_height": 50}):
        view = ImageMobjectFromCamera(Camera())
        view.stretch_to_fit_width(4).stretch_to_fit_height(2)
        primary_camera = MultiCamera([view])
        renderer = CairoRenderer(camera=primary_camera)
        main_target = renderer._target

        renderer.update_frame(None, mobjects=[view])
        first_target = renderer._sub_targets[id(view)]
        assert first_target.settings.pixel_width == max(
            1,
            int(100 * view.width / primary_camera.frame_width),
        )
        assert first_target.settings.pixel_height == max(
            1,
            int(50 * view.height / primary_camera.frame_height),
        )

        view.stretch_to_fit_width(6)
        renderer.update_frame(None, mobjects=[view])
        second_target = renderer._sub_targets[id(view)]
        assert second_target is not first_target
        with pytest.raises(RuntimeError, match="closed"):
            _ = first_target.pixels

        renderer.close()
        renderer.close()
        assert renderer._sub_targets == {}
        with pytest.raises(RuntimeError, match="closed"):
            _ = main_target.pixels
        with pytest.raises(RuntimeError, match="closed"):
            _ = second_target.pixels


@pytest.mark.parametrize("pixel_shape", [(128, 128), (96, 160)])
def test_nested_view_preserves_square_geometry(pixel_shape):
    width, height = pixel_shape
    with tempconfig({"pixel_width": width, "pixel_height": height}):
        view = ImageMobjectFromCamera(Camera())
        view.set(width=8)
        renderer = CairoRenderer(camera=MultiCamera([view]))
        try:
            renderer.render_mobjects(
                [Square(fill_color="#ffffff", fill_opacity=1, stroke_width=0), view]
            )
            pixels = renderer._sub_targets[id(view)].read_pixels()
            rows, columns = np.where(pixels[:, :, 0] > 128)
            assert len(rows) > 0
            assert np.ptp(columns) == pytest.approx(np.ptp(rows), abs=1)
        finally:
            renderer.close()


def test_removed_nested_views_release_their_target_subtree():
    with tempconfig({"pixel_width": 64, "pixel_height": 32}):
        leaf = ImageMobjectFromCamera(Camera())
        branch = ImageMobjectFromCamera(MultiCamera([leaf]))
        retained = ImageMobjectFromCamera(Camera())
        camera = MultiCamera([branch, retained])
        renderer = CairoRenderer(camera=camera)
        try:
            renderer.render_mobjects([branch, retained])
            branch_target = renderer._sub_targets[id(branch)]
            leaf_target = renderer._sub_targets[id(leaf)]
            retained_target = renderer._sub_targets[id(retained)]

            camera.image_mobjects_from_cameras.remove(branch)
            renderer.render_mobjects([retained])

            assert renderer._sub_targets == {id(retained): retained_target}
            for retired in (branch_target, leaf_target):
                with pytest.raises(RuntimeError, match="closed"):
                    _ = retired.pixels
            assert retained_target.pixels.size > 0
        finally:
            renderer.close()


def test_failed_nested_draw_releases_unfinished_target():
    with tempconfig({"pixel_width": 64, "pixel_height": 32}):
        view = ImageMobjectFromCamera(Camera())
        renderer = CairoRenderer(camera=MultiCamera([view]))
        try:
            renderer.render_mobjects([view])
            target = renderer._sub_targets[id(view)]
            with (
                patch.object(
                    _CairoDrawingContext, "draw", side_effect=ValueError("draw failed")
                ),
                pytest.raises(ValueError, match="draw failed"),
            ):
                renderer.render_mobjects([view])

            assert renderer._sub_targets == {}
            with pytest.raises(RuntimeError, match="closed"):
                _ = target.pixels
            renderer.render_mobjects([view])
            assert renderer._sub_targets[id(view)].pixels.size > 0
        finally:
            renderer.close()


@pytest.mark.parametrize(
    "operation",
    [
        lambda renderer: renderer.render_mobjects([]),
        lambda renderer: renderer.update_frame(None),
        lambda renderer: renderer.update_frame(None, ignore_skipping=False),
        lambda renderer: renderer.get_frame(),
        lambda renderer: renderer.get_image(),
        lambda renderer: renderer.save_static_frame_data(
            SimpleNamespace(moving_mobjects=[]), []
        ),
    ],
    ids=["draw", "update", "skipped-update", "pixels", "image", "static-frame"],
)
def test_closed_renderer_rejects_drawing(operation):
    with tempconfig({"pixel_width": 8, "pixel_height": 4}):
        renderer = CairoRenderer(skip_animations=True)
        renderer.close()
        with pytest.raises(RuntimeError, match="closed"):
            operation(renderer)


def test_renderer_close_releases_static_pixels():
    with tempconfig({"pixel_width": 32, "pixel_height": 18}):
        renderer = CairoRenderer()
        try:
            renderer.save_static_frame_data(
                SimpleNamespace(moving_mobjects=[]), [Square()]
            )
            snapshot = weakref.ref(renderer.static_image)
            renderer.close()
            gc.collect()
            assert renderer.static_image is None
            assert snapshot() is None
        finally:
            renderer.close()


def test_nested_views_disable_unsafe_static_reuse():
    view = ImageMobjectFromCamera(Camera())
    renderer = CairoRenderer(camera=MultiCamera([view]))
    scene = SimpleNamespace(moving_mobjects=[Square()])

    try:
        assert renderer.save_static_frame_data(scene, [view]) is None
        assert renderer._render_all_mobjects is True
    finally:
        renderer.close()


def test_multicamera_reports_recursive_view_controls_without_recursing_cycles():
    three_d_camera = ThreeDCamera()
    nested_camera = MultiCamera([ImageMobjectFromCamera(three_d_camera)])
    primary_camera = MultiCamera([ImageMobjectFromCamera(nested_camera)])
    nested_camera.add_image_mobject_from_camera(ImageMobjectFromCamera(primary_camera))

    indicators = primary_camera.get_mobjects_indicating_movement()

    assert primary_camera.frame in indicators
    assert nested_camera.frame in indicators
    assert three_d_camera.theta_tracker in indicators
    assert three_d_camera.zoom_tracker in indicators


def test_nested_camera_cycles_are_rejected():
    first = MultiCamera()
    second = MultiCamera()
    second_view = ImageMobjectFromCamera(second)
    first_view = ImageMobjectFromCamera(first)
    first.add_image_mobject_from_camera(second_view)
    second.add_image_mobject_from_camera(first_view)
    renderer = CairoRenderer(camera=first)

    try:
        with pytest.raises(RuntimeError, match="composition cycle"):
            renderer.render_mobjects([first_view, second_view])
    finally:
        renderer.close()
