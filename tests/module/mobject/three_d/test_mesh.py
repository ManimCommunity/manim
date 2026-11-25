import numpy as np
import pytest

from manim import (
    BLUE,
    RED,
    WHITE,
    ParametricSurface,
    Scene,
    Square,
    Transform,
    tempconfig,
)
from manim.material import PbrMaterial, UnlitMaterial
from manim.renderer.protocol import CapabilityError
from manim.utils.versioned_fields import field_version


def _saddle(resolution=(4, 3), **kwargs):
    return ParametricSurface(
        lambda u, v: [u, v, u * v],
        u_range=[-1, 1],
        v_range=[0, 3],
        resolution=resolution,
        **kwargs,
    )


def _uv_area(surface):
    corners = surface.points[surface.triangles][..., :2]
    (ux, uy), (vx, vy) = (
        (corners[:, 1] - corners[:, 0]).T,
        (corners[:, 2] - corners[:, 0]).T,
    )
    return (ux * vy - uy * vx) / 2


@pytest.mark.parametrize("flat", [False, True])
def test_surface_triangles_tile_the_parameter_domain(flat):
    surface = _saddle(flat=flat)

    np.testing.assert_allclose(
        surface.points[:, 2], surface.points[:, 0] * surface.points[:, 1]
    )
    # Same orientation and no overlap: the areas add up to the domain's.
    assert (_uv_area(surface) > 0).all()
    assert _uv_area(surface).sum() == pytest.approx(2 * 3)
    assert len(surface.triangles) == 2 * 4 * 3


def test_checkerboard_alternates_between_neighbouring_quads():
    surface = _saddle().set_fill_by_checkerboard(RED, BLUE)

    # Four vertices per quad, quads in (u, v) order; all of a quad's are its colour.
    colors = surface.rgbas[:, :3].reshape(4, 3, 4, 3)
    assert (colors == colors[:, :, :1]).all()
    is_red = np.isclose(colors[:, :, 0], RED.to_rgb()).all(axis=-1)
    np.testing.assert_array_equal(is_red, np.add.outer(range(4), range(3)) % 2 == 0)


def test_color_by_xyz_func_maps_values_onto_the_colorscale():
    surface = _saddle().set_color_by_xyz_func(lambda p: p[:, 2], [BLUE, WHITE], 0, 3)

    weights = np.clip(surface.points[:, 2] / 3, 0, 1)[:, None]
    expected = (1 - weights) * BLUE.to_rgb() + weights * WHITE.to_rgb()
    np.testing.assert_allclose(surface.rgbas[:, :3], expected, atol=1e-6)


def test_partial_surface_keeps_triangles_and_stops_at_the_given_proportion():
    surface = _saddle()
    partial = surface.copy().pointwise_become_partial(surface, 0, 0.5)

    np.testing.assert_array_equal(partial.triangles, surface.triangles)
    assert partial.points[:, 1].max() == pytest.approx(1.5)
    assert partial.points[:, 1].min() == pytest.approx(0)


def test_surfaces_of_different_resolution_cannot_be_aligned():
    with pytest.raises(ValueError, match="same resolution"):
        _saddle().align_points(_saddle(resolution=(5, 3)))


def test_wireframe_follows_the_parameter_lines():
    wireframe = _saddle().get_wireframe()

    assert len(wireframe) == (4 + 1) + (3 + 1)
    np.testing.assert_allclose(wireframe[0].get_start(), [-1, 0, 0])
    np.testing.assert_allclose(wireframe[0].get_end(), [-1, 3, -3])


def _moved_and_deformed(surface):
    surface.shift([0, 0, 3])
    # A change no placement can undo.
    points = np.array(surface.points)
    points[:, 2] += np.sin(points[:, 0])
    surface.points = points
    return surface


@pytest.mark.parametrize("textured", [False, True])
def test_checkerboard_keeps_the_points_of_a_moved_and_deformed_surface(textured):
    texture = np.full((2, 2, 4), 255, dtype=np.uint8) if textured else None
    surface = _moved_and_deformed(_saddle(texture=texture))
    before = np.array(surface.points).reshape(5, 4, 3)
    uvs = np.array(surface.uvs)

    surface.set_fill_by_checkerboard(RED, BLUE)

    quads = surface.points.reshape(4, 3, 4, 3)
    np.testing.assert_array_equal(quads[:, :, 0], before[:-1, :-1])
    np.testing.assert_array_equal(quads[:, :, 1], before[1:, :-1])
    np.testing.assert_array_equal(quads[:, :, 2], before[:-1, 1:])
    np.testing.assert_array_equal(quads[:, :, 3], before[1:, 1:])
    if textured:
        # Every corner keeps the texture coordinates of its grid point.
        np.testing.assert_array_equal(
            surface.uvs.reshape(4, 3, 4, 2)[:, :, 0], uvs.reshape(5, 4, 2)[:-1, :-1]
        )


@pytest.mark.parametrize("flat", [False, True])
def test_wireframe_goes_through_the_current_points(flat):
    surface = _moved_and_deformed(_saddle(flat=flat))

    wireframe = surface.get_wireframe()

    assert len(wireframe) == (4 + 1) + (3 + 1)
    np.testing.assert_allclose(wireframe[0].get_start(), [-1, 0, 3 + np.sin(-1)])
    np.testing.assert_allclose(wireframe[0].get_end(), [-1, 3, np.sin(-1)])
    np.testing.assert_allclose(wireframe[-1].get_start(), [-1, 3, np.sin(-1)])
    np.testing.assert_allclose(wireframe[-1].get_end(), [1, 3, 6 + np.sin(1)])


def test_wireframe_of_points_that_are_no_grid_raises():
    surface = _saddle()
    surface.points = surface.points[:-1]

    with pytest.raises(ValueError, match="resolution"):
        surface.get_wireframe()


def test_paths_and_meshes_cannot_be_transformed_into_each_other():
    with pytest.raises(ValueError, match="only into another vectorized mobject"):
        Transform(Square(), _saddle()).begin()
    with pytest.raises(ValueError, match="same triangles"):
        Transform(_saddle(), Square()).begin()


def test_interpolating_equal_colors_keeps_their_version():
    start = _saddle()
    end = start.copy().shift([1, 0, 0])
    mesh = start.copy()
    mesh.interpolate(start, end, 0.3)
    version = field_version(mesh, "rgbas")

    mesh.interpolate(start, end, 0.6)

    assert field_version(mesh, "rgbas") == version


def test_cairo_cannot_draw_meshes_yet():
    with tempconfig({"dry_run": True}):
        scene = Scene()
        scene.add(Square(), _saddle())
        with pytest.raises(CapabilityError, match="opengl"):
            scene.renderer.update_frame(scene)


def _saddles(start_material, end_material):
    start, end = _saddle(), _saddle()
    start.material, end.material = start_material, end_material
    return start, end, start.copy()


def test_pbr_materials_interpolate_and_end_on_the_target():
    target = PbrMaterial(roughness=1.0, metallic=0.5, emissive=(0.2, 0, 0))
    start, end, moving = _saddles(PbrMaterial(roughness=0.2), target)

    moving.interpolate(start, end, 0.5)
    assert moving.material == PbrMaterial(
        roughness=0.6, metallic=0.25, emissive=(0.1, 0, 0)
    )

    moving.interpolate(start, end, 1)
    assert moving.material is target


@pytest.mark.parametrize(
    ("start_material", "end_material"),
    [(UnlitMaterial(), PbrMaterial()), (PbrMaterial(), UnlitMaterial())],
    ids=["unlit to pbr", "pbr to unlit"],
)
def test_lit_and_unlit_switch_when_the_progress_reaches_one(
    start_material, end_material
):
    start, end, moving = _saddles(start_material, end_material)

    moving.interpolate(start, end, 0.99)
    assert moving.material == start_material
    moving.interpolate(start, end, 1)
    assert moving.material == end_material


def test_there_and_back_ends_on_the_start_material():
    from manim import Transform, there_and_back

    start, end, _ = _saddles(UnlitMaterial(), PbrMaterial())
    animation = Transform(start, end, rate_func=there_and_back)
    animation.begin()
    for alpha in (0.25, 0.5, 0.75, 1):
        animation.interpolate(alpha)
    assert start.material == UnlitMaterial()


def test_meshes_are_lit_by_default():
    assert _saddle().material == PbrMaterial()
