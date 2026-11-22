"""Both renderers draw meshes: OpenGL with depth testing and lighting per pixel,
Cairo as faces sorted far to near, lit per face by the same function.
"""

import numpy as np
import pytest

from manim import (
    BLUE,
    DEGREES,
    OUT,
    PI,
    PURE_BLUE,
    PURE_RED,
    RIGHT,
    TAU,
    ImageMobject,
    MeshMobject,
    ParametricSurface,
    Scene,
    Square,
    ThreeDScene,
    config,
    tempconfig,
)
from manim.appearance.material import PbrMaterial, UnlitMaterial

WIDTH, HEIGHT = 384, 216
RENDERERS = pytest.mark.parametrize("renderer", ["cairo", "opengl"])


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


def _plane(z, color, resolution=4, **kwargs):
    return ParametricSurface(
        lambda u, v: [u, v, z],
        u_range=[-3, 3],
        v_range=[-2, 2],
        resolution=resolution,
        color=color,
        material=UnlitMaterial(),
        **kwargs,
    )


def _sphere(radius=2, resolution=(48, 24), **kwargs):
    return ParametricSurface(
        lambda u, v: (
            radius
            * np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), -np.cos(v)])
        ),
        u_range=[0, TAU],
        v_range=[0, np.pi],
        resolution=resolution,
        **kwargs,
    )


@RENDERERS
def test_texture_is_drawn_like_an_image(renderer):
    # u runs left to right across the image, v from its bottom to its top.
    texture = np.zeros((40, 60, 4), dtype=np.uint8)
    texture[..., 0] = np.linspace(0, 255, 60)[None, :]
    texture[..., 2] = np.linspace(255, 0, 40)[:, None]
    texture[..., 3] = 255

    # Cairo colors each face by the texture at its middle.
    surface, _ = _image(
        renderer, lambda: [_plane(0, BLUE, resolution=(60, 40), texture=texture)]
    )
    image, _ = _image(
        "cairo",
        lambda: [
            ImageMobject(texture).stretch_to_fit_width(6).stretch_to_fit_height(4)
        ],
    )

    assert np.abs(surface - image).mean() < 0.01


@RENDERERS
def test_opaque_meshes_hide_what_is_behind_them(renderer):
    front = _plane(1, PURE_RED)
    back = _plane(-1, PURE_BLUE)

    pixels, _ = _image(renderer, lambda: [front, back], ThreeDScene)

    np.testing.assert_allclose(_pixel(pixels, 0, 0), [1, 0, 0])


@RENDERERS
@pytest.mark.parametrize("scene_class", [Scene, ThreeDScene])
def test_faces_of_a_mesh_hide_those_behind_them(renderer, scene_class):
    # The sphere's near half is red, its far half blue. Turned over, its faces come
    # near to far.
    sphere = _sphere(1, material=UnlitMaterial()).rotate(PI, RIGHT)
    near = sphere.points[:, 2] > 0
    rgbas = np.zeros((len(near), 4))
    rgbas[:, 0], rgbas[:, 2], rgbas[:, 3] = near, ~near, 1
    sphere.set_vertex_colors(rgbas)

    pixels, _ = _image(renderer, lambda: [sphere], scene_class)

    # Off the pole, where on Cairo many faces meet.
    np.testing.assert_allclose(_pixel(pixels, 0.4, 0.3), [1, 0, 0], atol=0.05)


def test_translucent_meshes_show_only_their_front_layer(using_opengl_renderer):
    sphere = _sphere(opacity=0.5, material=UnlitMaterial())
    top = sphere.points[:, 2] > 0
    rgbas = np.zeros((len(top), 4))
    rgbas[:, 0], rgbas[:, 2], rgbas[:, 3] = top, ~top, 0.5
    sphere.set_vertex_colors(rgbas)

    pixels, _ = _image("opengl", lambda: [sphere], ThreeDScene)

    np.testing.assert_allclose(_pixel(pixels, 0.5, 0.5), [0.5, 0, 0], atol=0.01)


@RENDERERS
def test_meshes_are_lit_from_the_light_source(renderer):
    pixels, camera = _image(
        renderer,
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


@RENDERERS
def test_meshes_fixed_in_frame_ignore_the_camera(renderer):
    plane = _plane(0, PURE_RED).fix_in_frame()

    pixels, _ = _image(renderer, lambda: [plane], ThreeDScene, phi=80 * DEGREES)

    np.testing.assert_allclose(_pixel(pixels, 2.5, 1.5), [1, 0, 0])


@RENDERERS
def test_meshes_are_not_lit_under_a_flat_camera(renderer):
    pixels, _ = _image(renderer, lambda: [_sphere(color=PURE_RED)])

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


@RENDERERS
def test_faces_are_lit_as_the_lighting_function(renderer):
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
        renderer, lambda: [plane], ThreeDScene, phi=40 * DEGREES, theta=-60 * DEGREES
    )

    rotation = camera.get_rotation_matrix()
    eye = camera.frame_center + rotation.T @ [0, 0, camera.get_focal_distance()]
    expected = shade(
        np.array([plane.rgbas[0, :3]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.zeros((1, 3)),
        eye,
        [Light(camera.light_source.get_center())],
        AMBIENT,
        material,
    )
    # Cairo lights each face at its middle, so allow for the face's size.
    np.testing.assert_allclose(pixels[HEIGHT // 2, WIDTH // 2], expected[0], atol=0.05)


def test_a_material_change_redraws_on_cairo():
    from manim.renderer.cairo import CairoRenderer

    with tempconfig({"pixel_width": 64, "pixel_height": 36}):
        scene = ThreeDScene()
        sphere = _sphere(color=BLUE)
        renderer = CairoRenderer(camera=scene.camera)
        fresh = CairoRenderer(camera=scene.camera)
        try:
            renderer.render_mobjects([sphere])
            renderer.render_mobjects([sphere])
            sphere.material = PbrMaterial(roughness=1, metallic=1)
            renderer.render_mobjects([sphere])
            fresh.render_mobjects([sphere])
            np.testing.assert_array_equal(renderer.get_frame(), fresh.get_frame())
        finally:
            renderer.close()
            fresh.close()


def test_face_keys_follow_every_change_of_the_mesh():
    from manim.renderer.cairo._mesh_faces import mesh_faces
    from manim.renderer.extract import snapshot

    camera = ThreeDScene().camera
    sphere = _sphere(color=BLUE, resolution=(4, 2))

    def keys():
        return [face.key for face in mesh_faces(snapshot(sphere), camera._state())]

    before = keys()
    assert keys() == before
    sphere.material = PbrMaterial(roughness=1, metallic=1)
    after = keys()
    assert not set(after) & set(before)


def test_faces_are_sorted_by_their_own_middle():
    from manim.renderer.cairo._mesh_faces import mesh_faces
    from manim.renderer.extract import snapshot

    # A tetrahedron around the origin: the middle of every face's bounding box is
    # the origin, the mean of its corners is not.
    corners = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    mesh = MeshMobject(
        points=corners, triangles=np.array([[0, 1, 2], [3, 0, 2], [0, 1, 3], [3, 1, 2]])
    )
    references = [
        face.payload.depth_reference
        for face in mesh_faces(snapshot(mesh), ThreeDScene().camera._state())
    ]
    assert len({tuple(np.round(r, 6)) for r in references}) == 4


def test_face_outlines_look_alike_on_both_renderers(using_opengl_renderer):
    def surface():
        plane = _plane(0, PURE_BLUE, resolution=(6, 4))
        return [plane.set_stroke(PURE_RED, width=4)]

    cairo, _ = _image("cairo", surface)
    opengl, _ = _image("opengl", surface)

    # Inside the surface the outlines land on the same pixels. (At its border,
    # OpenGL draws only the half of the outline that lies on the surface. Cairo
    # strokes each face's outline, so a side shared by two faces is drawn twice and
    # comes out darker; OpenGL draws it once, half in each face.)
    inside = np.s_[
        int((0.5 - 1.9 / config.frame_height) * HEIGHT) : int(
            (0.5 + 1.9 / config.frame_height) * HEIGHT
        ),
        int((0.5 - 2.9 / config.frame_width) * WIDTH) : int(
            (0.5 + 2.9 / config.frame_width) * WIDTH
        ),
        0,
    ]
    outlined_cairo, outlined_opengl = cairo[inside] > 0.25, opengl[inside] > 0.25
    assert outlined_cairo.sum() > 0
    assert (outlined_cairo != outlined_opengl).mean() < 0.005


@RENDERERS
def test_meshes_are_sorted_with_3d_paths(renderer):
    # Added before the plane, but in front of it: drawn after it.
    square = Square(2, fill_opacity=1, color=PURE_RED).shift(OUT)
    square.set_shade_in_3d(True)
    plane = _plane(0, PURE_BLUE)

    pixels, _ = _image(renderer, lambda: [square, plane], ThreeDScene)

    assert _pixel(pixels, 0, 0)[0] > 0.9
