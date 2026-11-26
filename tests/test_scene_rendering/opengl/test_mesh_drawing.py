"""The OpenGL renderer draws meshes with textures, depth testing and lighting."""

import numpy as np

from manim import (
    BLUE,
    DEGREES,
    PURE_BLUE,
    PURE_RED,
    TAU,
    ImageMobject,
    ParametricSurface,
    Scene,
    ThreeDScene,
    config,
    tempconfig,
)
from manim.appearance.material import PbrMaterial, UnlitMaterial

WIDTH, HEIGHT = 384, 216


def _image(renderer, mobjects, scene_class=Scene, **orientation):
    with tempconfig(
        {
            "renderer": renderer,
            "dry_run": True,
            "pixel_width": WIDTH,
            "pixel_height": HEIGHT,
        }
    ):
        scene = scene_class()
        try:
            if orientation:
                scene.set_camera_orientation(**orientation)
            scene.add(*mobjects())
            pixels = np.asarray(scene.get_image().convert("RGB"), dtype=float) / 255
            camera = scene.renderer.camera
        finally:
            scene.renderer.close()
    return pixels, camera


def _pixel(pixels, x, y):
    """The pixel at scene coordinates ``(x, y)`` seen from the default camera."""
    column = int((x / config.frame_width + 0.5) * WIDTH)
    row = int((0.5 - y / config.frame_height) * HEIGHT)
    return pixels[row, column]


def _plane(z, color, **kwargs):
    return ParametricSurface(
        lambda u, v: [u, v, z],
        u_range=[-3, 3],
        v_range=[-2, 2],
        resolution=4,
        color=color,
        material=UnlitMaterial(),
        **kwargs,
    )


def _sphere(radius=2, **kwargs):
    return ParametricSurface(
        lambda u, v: (
            radius
            * np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), -np.cos(v)])
        ),
        u_range=[0, TAU],
        v_range=[0, np.pi],
        resolution=(48, 24),
        **kwargs,
    )


def test_texture_is_drawn_like_an_image(using_opengl_renderer):
    # u runs left to right across the image, v from its bottom to its top.
    texture = np.zeros((40, 60, 4), dtype=np.uint8)
    texture[..., 0] = np.linspace(0, 255, 60)[None, :]
    texture[..., 2] = np.linspace(255, 0, 40)[:, None]
    texture[..., 3] = 255

    surface, _ = _image("opengl", lambda: [_plane(0, BLUE, texture=texture)])
    image, _ = _image(
        "cairo",
        lambda: [
            ImageMobject(texture).stretch_to_fit_width(6).stretch_to_fit_height(4)
        ],
    )

    assert np.abs(surface - image).mean() < 0.01


def test_opaque_meshes_hide_what_is_behind_them(using_opengl_renderer):
    front = _plane(1, PURE_RED)
    back = _plane(-1, PURE_BLUE)

    pixels, _ = _image("opengl", lambda: [front, back], ThreeDScene)

    np.testing.assert_allclose(_pixel(pixels, 0, 0), [1, 0, 0])


def test_translucent_meshes_show_only_their_front_layer(using_opengl_renderer):
    sphere = _sphere(opacity=0.5, material=UnlitMaterial())
    top = sphere.points[:, 2] > 0
    rgbas = np.zeros((len(top), 4))
    rgbas[:, 0], rgbas[:, 2], rgbas[:, 3] = top, ~top, 0.5
    sphere.set_vertex_colors(rgbas)

    pixels, _ = _image("opengl", lambda: [sphere], ThreeDScene)

    np.testing.assert_allclose(_pixel(pixels, 0.5, 0.5), [0.5, 0, 0], atol=0.01)


def test_meshes_are_lit_from_the_light_source(using_opengl_renderer):
    pixels, camera = _image(
        "opengl",
        lambda: [_sphere(material=PbrMaterial(roughness=1))],
        ThreeDScene,
        phi=60 * DEGREES,
        theta=30 * DEGREES,
    )

    # Seen from the camera, the light is above the sphere in this direction.
    light = camera.project_point(camera.light_source.get_center())[:2]
    toward = 1.5 * light / np.linalg.norm(light)
    lit, unlit = _pixel(pixels, *toward), _pixel(pixels, *-toward)

    assert lit.sum() > 1.3 * unlit.sum()


def test_meshes_fixed_in_frame_ignore_the_camera(using_opengl_renderer):
    plane = _plane(0, PURE_RED).fix_in_frame()

    pixels, _ = _image("opengl", lambda: [plane], ThreeDScene, phi=80 * DEGREES)

    np.testing.assert_allclose(_pixel(pixels, 2.5, 1.5), [1, 0, 0])


def test_meshes_are_not_lit_under_a_flat_camera(using_opengl_renderer):
    pixels, _ = _image("opengl", lambda: [_sphere(color=PURE_RED)])

    np.testing.assert_allclose(_pixel(pixels, 0, 0), [1, 0, 0], atol=0.01)


def test_the_mesh_shader_shades_as_the_lighting_function(using_opengl_renderer):
    from manim.renderer._lighting import AMBIENT, Light, shade

    material = PbrMaterial(roughness=0.4, metallic=0.3)
    plane = ParametricSurface(
        lambda u, v: [u, v, 0],
        u_range=[-3, 3],
        v_range=[-2, 2],
        resolution=8,
        color=BLUE,
        material=material,
    )
    pixels, camera = _image(
        "opengl", lambda: [plane], ThreeDScene, phi=40 * DEGREES, theta=-60 * DEGREES
    )

    # The centre of the plane is drawn at the centre of the frame.
    rotation = camera.get_rotation_matrix()
    eye = camera.frame_center + rotation.T @ [0, 0, camera.get_focal_distance()]
    light = Light(camera.light_source.get_center())
    expected = shade(
        np.array([plane.rgbas[0, :3]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.zeros((1, 3)),
        eye,
        [light],
        AMBIENT,
        material,
    )
    np.testing.assert_allclose(pixels[HEIGHT // 2, WIDTH // 2], expected[0], atol=0.02)


def test_the_mesh_shader_draws_the_highlight_of_the_lighting_function(
    using_opengl_renderer,
):
    from manim.renderer._lighting import AMBIENT, Light, shade

    material = PbrMaterial(roughness=0.3, metallic=0.0)
    plane = ParametricSurface(
        lambda u, v: [u, v, 0],
        u_range=[-6, 6],
        v_range=[-4, 4],
        resolution=8,
        color=BLUE,
        material=material,
    )
    pixels, camera = _image("opengl", lambda: [plane], ThreeDScene)
    # Seen straight from above: the highlight is where the half vector is the normal.
    eye = camera.frame_center + [0, 0, camera.get_focal_distance()]
    light = Light(camera.light_source.get_center())
    xs = np.linspace(-5, 5, 11)
    points = np.array([[x, y, 0.0] for x in xs for y in np.linspace(-3, 3, 7)])
    expected = shade(
        np.repeat(plane.rgbas[:1, :3], len(points), axis=0),
        np.repeat([[0.0, 0.0, 1.0]], len(points), axis=0),
        points,
        eye,
        [light],
        AMBIENT,
        material,
    )
    drawn = np.array([_pixel(pixels, x, y) for x, y, _ in points])
    assert np.ptp(expected[:, 2]) > 0.1  # the highlight shows
    np.testing.assert_allclose(drawn, expected, atol=0.03)


def test_rough_meshes_have_no_holes_on_their_dark_side(using_opengl_renderer):
    # At roughness 1 the highlight's exponent is 0; pow(0, 0) is undefined in GLSL
    # and was black on some drivers where the surface turns away from the light.
    pixels, camera = _image(
        "opengl",
        lambda: [_sphere(material=PbrMaterial(roughness=1))],
        ThreeDScene,
        phi=60 * DEGREES,
        theta=30 * DEGREES,
    )
    light = camera.project_point(camera.light_source.get_center())[:2]
    away = -light / np.linalg.norm(light)
    for radius in (0.5, 1.0, 1.5, 1.8):
        assert _pixel(pixels, *(radius * away)).sum() > 0.3
