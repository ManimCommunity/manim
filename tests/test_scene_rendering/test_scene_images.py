"""Current-state inspection is independent of animation and movie emission."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from PIL import Image

from manim import RIGHT, Scene, Square, ThreeDScene, ZoomedScene, tempconfig


@pytest.fixture(params=["cairo", "opengl"])
def image_scene(request):
    with tempconfig(
        {
            "renderer": request.param,
            "dry_run": True,
            "pixel_width": 128,
            "pixel_height": 128,
            "frame_rate": 4,
        }
    ):
        scene = Scene()
        yield scene
        if request.param == "cairo":
            scene.renderer.close()
        else:
            scene.renderer.context.release()


def test_fresh_image_without_evaluation(image_scene, monkeypatch):
    scene = image_scene
    calls = []
    square = Square(fill_opacity=1, stroke_width=0)
    square.add_updater(lambda m, dt: calls.append(dt))
    scene.add(square)
    calls.clear()

    def unexpected(*args, **kwargs):
        pytest.fail("Image inspection must not execute or emit movie frames")

    monkeypatch.setattr(scene, "construct", unexpected)
    monkeypatch.setattr(scene.renderer.file_writer, "write_frame", unexpected)
    scene.renderer.update_frame(scene)
    old = scene.renderer.get_frame()
    square.shift(3 * RIGHT)
    image = scene.get_image()
    assert isinstance(image, Image.Image)
    assert image.size == (128, 128)
    assert not np.array_equal(old, np.asarray(image))
    np.testing.assert_array_equal(scene.renderer.get_frame(), old)
    assert calls == []
    assert scene.time == 0
    assert scene.renderer.num_plays == 0
    square.shift(-3 * RIGHT)
    np.testing.assert_array_equal(np.asarray(scene.get_image()), old)


def test_image_uses_existing_resolution(image_scene):
    image_scene.add(Square())
    with tempconfig({"pixel_width": 72, "pixel_height": 40}):
        assert image_scene.get_image().size == (128, 128)


def test_show_uses_fresh_image(image_scene, monkeypatch):
    images = []
    monkeypatch.setattr(Image.Image, "show", lambda image: images.append(image))
    image_scene.add(Square())
    image_scene.show()
    assert len(images) == 1
    np.testing.assert_array_equal(images[0], image_scene.get_image())


def test_capture_during_construct_and_after_render(image_scene, monkeypatch):
    images = []

    def construct():
        square = Square(fill_opacity=1)
        image_scene.add(square)
        images.append(image_scene.get_image())
        image_scene.play(square.animate.shift(3 * RIGHT))
        clock = image_scene.time
        images.append(image_scene.get_image())
        assert image_scene.time == clock
        assert image_scene.renderer.num_plays == 1

    monkeypatch.setattr(image_scene, "construct", construct)
    image_scene.render()
    assert not np.array_equal(images[0], images[1])
    np.testing.assert_array_equal(images[1], image_scene.get_image())


def test_cairo_image_after_close_and_with_static_cache():
    with tempconfig({"dry_run": True, "pixel_width": 128, "pixel_height": 128}):
        scene = Scene()
        scene.add(Square(fill_opacity=1))
        renderer = scene.renderer
        renderer.static_image = np.full((128, 128, 4), 77, dtype=np.uint8)
        static = renderer.static_image
        image = scene.get_image()
        assert renderer.static_image is static
        renderer.close()
        np.testing.assert_array_equal(image, scene.get_image())
        assert renderer._closed
        assert renderer._target._pixels.size == 0


@pytest.mark.parametrize("scene_class", [Scene, ThreeDScene, ZoomedScene])
def test_cairo_snapshot_camera_and_nested_view_parity(scene_class):
    with tempconfig({"dry_run": True, "pixel_width": 128, "pixel_height": 128}):
        scene = scene_class()
        try:
            scene.setup()
            scene.add(Square(fill_opacity=1))
            if isinstance(scene, ZoomedScene):
                scene.activate_zooming(animate=False)
            scene.camera.frame.shift(RIGHT)
            image = scene.get_image()
            scene.renderer.update_frame(scene)
            np.testing.assert_array_equal(image, scene.renderer.get_frame())
        finally:
            scene.renderer.close()


def test_opengl_snapshot_restores_target_on_failure(image_scene, monkeypatch):
    renderer = image_scene.renderer
    if not hasattr(renderer, "context"):
        pytest.skip("OpenGL-specific resource test")
    target = renderer.frame_buffer_object
    viewport = renderer.context.viewport
    elapsed = renderer.animation_elapsed_time
    original = renderer._draw_scene

    def fail(scene):
        raise ValueError("drawing failed")

    monkeypatch.setattr(renderer, "_draw_scene", fail)
    with pytest.raises(ValueError, match="drawing failed"):
        image_scene.get_image()
    assert renderer.frame_buffer_object is target
    assert renderer.context.fbo is target
    assert renderer.context.viewport == viewport
    assert renderer.animation_elapsed_time == elapsed
    monkeypatch.setattr(renderer, "_draw_scene", original)
    assert image_scene.get_image().size == (128, 128)


def test_opengl_snapshot_includes_meshes(image_scene):
    if not hasattr(image_scene.renderer, "context"):
        pytest.skip("OpenGL-specific mesh test")
    from manim.renderer.opengl.shader import Mesh, Shader

    shader = Shader(
        image_scene.renderer.context,
        source={
            "vertex_shader": """
            #version 330
            in vec3 point;
            void main() { gl_Position = vec4(point, 1.0); }
        """,
            "fragment_shader": """
            #version 330
            out vec4 color;
            void main() { color = vec4(1.0, 0.0, 0.0, 1.0); }
        """,
        },
    )
    attributes = np.zeros(3, dtype=[("point", np.float32, (3,))])
    attributes["point"] = [(-0.5, -0.5, 0), (0.5, -0.5, 0), (0, 0.5, 0)]
    mesh = Mesh(shader=shader, attributes=attributes)
    image_scene.add(mesh)
    pixels = np.asarray(image_scene.get_image())
    np.testing.assert_array_equal(pixels[64, 64, :3], [255, 0, 0])
    image_scene.renderer.update_frame(image_scene)
    np.testing.assert_array_equal(pixels, image_scene.renderer.get_frame())


def test_opengl_snapshot_requires_owner_thread(image_scene):
    if not hasattr(image_scene.renderer, "context"):
        pytest.skip("OpenGL-specific thread test")
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(image_scene.get_image)
        with pytest.raises(RuntimeError, match="render thread"):
            future.result()
