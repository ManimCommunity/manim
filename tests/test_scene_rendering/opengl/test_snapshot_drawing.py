"""The OpenGL renderer draws the common mobject classes like the Cairo renderer."""

import numpy as np
import pytest
from PIL import Image

from manim import (
    BLUE,
    DEGREES,
    DR,
    GREEN,
    LEFT,
    ORANGE,
    PI,
    PURE_BLUE,
    PURE_RED,
    RED,
    RESAMPLING_ALGORITHMS,
    RIGHT,
    UL,
    UP,
    UR,
    YELLOW,
    Annulus,
    Arc,
    Circle,
    ImageMobject,
    Line,
    MathTex,
    PMobject,
    Scene,
    Square,
    Text,
    ThreeDScene,
    Triangle,
    VMobject,
    ZoomedScene,
    tempconfig,
)


def _text():
    return [Text("Hello, snapshots", font_size=72)]


def _equation():
    return [MathTex(r"\int_0^1 x^2\,dx = \tfrac{1}{3}").scale(2)]


def _holes():
    return [Annulus(inner_radius=1, outer_radius=2, fill_opacity=0.8, color=BLUE)]


def _gradients():
    circle = Circle(2, fill_opacity=1, stroke_width=12)
    return [circle.set_fill([PURE_RED, PURE_BLUE]).set_stroke([GREEN, ORANGE])]


def _strokes():
    zigzag = VMobject(stroke_width=30).set_points_as_corners(
        [[-4, -1, 0], [-2, 1, 0], [0, -1, 0], [2, 1, 0], [4, -1, 0]]
    )
    return [zigzag]


def _open_subpaths():
    # Each open subpath is filled as if closed by a chord to its own start.
    path = VMobject(fill_opacity=1, stroke_width=0)
    path.points = np.concatenate(
        [
            Arc(1.5, angle=PI).shift(2.5 * LEFT).points,
            Arc(1.5, angle=PI).shift(2.5 * RIGHT + UP).points,
        ]
    )
    return [path]


def _thin_lines():
    return [
        Line([-6, y, 0], [6, y + 1, 0], stroke_width=1) for y in np.linspace(-3, 2, 6)
    ]


def _point_cloud():
    cloud = PMobject(stroke_width=6)
    xs = np.linspace(-5, 5, 60)
    cloud.add_points(np.c_[xs, np.sin(xs), np.zeros_like(xs)], color=YELLOW)
    return [cloud]


def _images():
    pixels = np.zeros((8, 12, 4), dtype=np.uint8)
    pixels[..., 0] = np.linspace(0, 255, 12)[None, :]
    pixels[..., 2] = np.linspace(255, 0, 8)[:, None]
    pixels[..., 3] = 255
    sharp = ImageMobject(pixels).set_resampling_algorithm(
        RESAMPLING_ALGORITHMS["nearest"]
    )
    smooth = ImageMobject(pixels)
    return [
        sharp.scale_to_fit_width(5).shift(3 * LEFT),
        smooth.scale_to_fit_width(5).shift(3 * RIGHT),
    ]


def _overlapping():
    # Drawn one after the other: the blue square must cover the red one.
    return [
        Square(3, fill_opacity=0.6, color=RED, stroke_width=6).shift(LEFT),
        Square(3, fill_opacity=0.6, color=BLUE, stroke_width=6).shift(RIGHT),
    ]


def _image(renderer, build, scene_class=Scene):
    with tempconfig(
        {"renderer": renderer, "dry_run": True, "pixel_width": 384, "pixel_height": 216}
    ):
        scene = scene_class()
        try:
            scene.add(*build(scene) if scene_class is not Scene else build())
            pixels = np.asarray(scene.get_image().convert("RGB"), dtype=float) / 255
        finally:
            scene.renderer.close()
    return pixels


@pytest.mark.parametrize(
    "build",
    [
        _text,
        _equation,
        _holes,
        _open_subpaths,
        _gradients,
        _strokes,
        _thin_lines,
        _overlapping,
        _point_cloud,
        _images,
    ],
    ids=lambda build: build.__name__.strip("_"),
)
def test_drawing_matches_cairo(using_opengl_renderer, build):
    cairo = _image("cairo", build)
    opengl = _image("opengl", build)

    assert cairo.sum() > 0
    # Within 5%: sub-pixel lines come out about 4% lighter than Cairo's.
    assert opengl.sum() == pytest.approx(cairo.sum(), rel=0.05)
    assert np.abs(opengl - cairo).mean() < 0.01


def _zoomed(scene):
    scene.setup()
    scene.zoomed_camera.frame.move_to(2 * LEFT)
    scene.zoomed_display.to_corner(DR)
    scene.activate_zooming()
    # Upright and filled, so that a view drawn upside down cannot match.
    return [Triangle(fill_opacity=1, stroke_width=0).scale(0.5).move_to(2 * LEFT)]


def _moving_camera_scene():
    # Any scene moves its camera frame; build() takes the scene, so this is not Scene.
    return Scene()


def _zoomed_scene():
    return ZoomedScene(zoomed_display_height=4, zoomed_display_width=6)


def _fixed_in_3d(scene):
    scene.set_camera_orientation(phi=65 * DEGREES, theta=-40 * DEGREES, zoom=0.8)
    scene.add_fixed_in_frame_mobjects(Text("title", font_size=48).to_corner(UL))
    scene.add_fixed_orientation_mobjects(Text("label").move_to([2, 2, 1]))
    return [Square(3, fill_opacity=0.5, color=BLUE)]


def _fixed_in_moved_frame(scene):
    scene.camera.frame.scale(0.5).shift(LEFT)
    return [
        Circle(2, color=RED, fill_opacity=0.4),
        Square(2, fill_opacity=1, stroke_width=0).to_corner(UR).fix_in_frame(),
    ]


@pytest.mark.parametrize(
    ("scene_class", "build"),
    [
        (_zoomed_scene, _zoomed),
        (ThreeDScene, _fixed_in_3d),
        (_moving_camera_scene, _fixed_in_moved_frame),
    ],
    ids=["camera_view", "fixed_in_3d", "fixed_in_moved_frame"],
)
def test_scenes_match_cairo(using_opengl_renderer, scene_class, build):
    cairo = _image("cairo", build, scene_class)
    opengl = _image("opengl", build, scene_class)

    assert opengl.sum() == pytest.approx(cairo.sum(), rel=0.05)
    assert np.abs(opengl - cairo).mean() < 0.01


def test_images_that_compute_their_pixels_are_redrawn(using_opengl_renderer):
    from manim.mobject.types.image_mobject import AbstractImageMobject

    class Flashing(AbstractImageMobject):
        def __init__(self):
            self.value = 0
            super().__init__(scale_to_resolution=0)

        def get_pixel_array(self):
            pixels = np.zeros((4, 4, 4), dtype=np.uint8)
            pixels[..., self.value] = pixels[..., 3] = 255
            return pixels

    with tempconfig(
        {"renderer": "opengl", "dry_run": True, "pixel_width": 64, "pixel_height": 36}
    ):
        scene = Scene()
        image = Flashing().scale(2)
        scene.add(image)
        try:
            # One renderer draws both images, with its texture cache.
            with scene.renderer._scene_images():
                red = np.asarray(scene.get_image())[18, 32, :3]
                image.value = 2
                blue = np.asarray(scene.get_image())[18, 32, :3]
        finally:
            scene.renderer.close()
    assert red[0] > 200
    assert red[2] < 50
    assert blue[2] > 200
    assert blue[0] < 50


def _red_over_blue(height, width):
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[: height // 2, :, 0] = 255
    pixels[height // 2 :, :, 2] = 255
    return pixels


@pytest.mark.parametrize("source", ["file", "array"])
def test_images_show_their_pixels(using_opengl_renderer, tmp_path, source):
    pixels = _red_over_blue(4, 6)
    if source == "file":
        path = tmp_path / "red_over_blue.png"
        Image.fromarray(_red_over_blue(64, 64)).save(path)
        image = ImageMobject(path).scale_to_fit_height(4)
    else:
        image = ImageMobject(pixels).scale_to_fit_height(4)
    scene = Scene()
    scene.add(image)
    frame = np.asarray(scene.get_image().convert("RGB"))
    height, width, _ = frame.shape
    # The centers of the image's upper and lower halves.
    np.testing.assert_array_equal(frame[int(height * 3 / 8), width // 2], [255, 0, 0])
    np.testing.assert_array_equal(frame[int(height * 5 / 8), width // 2], [0, 0, 255])


@pytest.mark.parametrize("gap", [1.1, 2.2, 2.3, 3.0])
def test_fills_that_share_a_group_are_all_drawn(gap):
    # Two squares that do not overlap are filled together, inside one rectangle of
    # pixels. Apple's software renderer (macOS without a GPU) lost one or both for
    # these distances when that rectangle was set with the scissor test.
    def build():
        return [
            Square(fill_opacity=1, color=RED, stroke_width=0).shift(gap * LEFT),
            Square(fill_opacity=1, color=BLUE, stroke_width=0).shift(gap * RIGHT),
        ]

    with tempconfig(
        {"renderer": "opengl", "dry_run": True, "pixel_width": 480, "pixel_height": 270}
    ):
        scene = Scene()
        try:
            scene.add(*build())
            pixels = np.asarray(scene.get_image().convert("RGB"))
        finally:
            scene.renderer.close()

    covered = pixels.max(axis=2) > 20
    # Each square is 2 of 8 units high: 67.5 pixels.
    assert covered[:, :240].sum() == pytest.approx(67.5**2, rel=0.03)
    assert covered[:, 240:].sum() == pytest.approx(67.5**2, rel=0.03)
