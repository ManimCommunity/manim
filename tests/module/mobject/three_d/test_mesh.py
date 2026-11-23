import numpy as np
import pytest

from manim import (
    BLUE,
    BLUE_D,
    GREEN,
    ORIGIN,
    PI,
    RED,
    RIGHT,
    UL,
    UP,
    WHITE,
    Arrow3D,
    Cone,
    Cube,
    Cylinder,
    Dot3D,
    Line3D,
    ManimColor,
    MeshMobject,
    Mobject,
    ParametricSurface,
    PolygonFace,
    Prism,
    Sphere,
    Square,
    Surface,
    Tetrahedron,
    ThreeDAxes,
    Torus,
    Transform,
    VGroup,
    VMobject,
)
from manim.appearance.material import PbrMaterial, UnlitMaterial
from manim.utils.space_ops import rotate_vector
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

    # Four vertices per quad, quads in (u, v) order; all of a quad's are its color.
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


def test_interpolating_equal_outlines_keeps_their_versions():
    start = _saddle(stroke_width=3.7)
    end = start.copy().shift([1, 0, 0])
    mesh = start.copy()
    mesh.interpolate(start, end, 0.2)
    versions = [field_version(mesh, name) for name in ("stroke_rgbas", "stroke_width")]

    mesh.interpolate(start, end, 0.6)

    assert mesh.stroke_width == 3.7
    assert versions == [
        field_version(mesh, name) for name in ("stroke_rgbas", "stroke_width")
    ]


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


@pytest.mark.parametrize("name", ["u_range", "v_range", "resolution"])
def test_parametric_surface_ranges_and_resolution_are_pairs(name):
    kwargs = {name: (0, 1, 2)}
    with pytest.raises(ValueError, match=f"{name} needs two values"):
        ParametricSurface(lambda u, v: np.array([u, v, 0]), **kwargs)
    surface = ParametricSurface(
        lambda u, v: np.array([u, v, 0]), u_range=[0, 2], resolution=[3, 4]
    )
    assert surface.u_range == (0, 2)
    assert surface.resolution == (3, 4)


def test_checkered_surface_colours_faces_by_value_at_their_outlines_middle():
    from manim.mobject.three_d.three_dimensions import _halfway_around

    axes = ThreeDAxes(x_range=[0, 4], y_range=[0, 4], z_range=[0, 4])
    surface = Surface(
        lambda u, v: axes.c2p(u, v, u * v / 4),
        u_range=[0, 4],
        v_range=[0, 4],
        resolution=4,
        checkerboard_colors=[RED, BLUE],
    )
    # Halfway around a face's outline, as get_midpoint of the face as a path.
    for corners, point in zip(
        surface.points[surface.get_faces()], _halfway_around(surface), strict=True
    ):
        face = VMobject().set_points_as_corners([*corners, corners[0]])
        np.testing.assert_allclose(point, face.get_midpoint(), atol=1e-9)

    surface.set_fill_by_value(axes, [(RED, 0), (WHITE, 4)], axis=2)

    values = axes.point_to_coords(_halfway_around(surface))[:, 2] / 4
    expected = (1 - values[:, None]) * RED.to_rgb() + values[:, None] * WHITE.to_rgb()
    np.testing.assert_allclose(surface.rgbas[::4, :3], expected, atol=1e-9)
    # The outlines take the faces' colors, as the faces' strokes did.
    np.testing.assert_allclose(surface.stroke_rgbas[:, :3], expected, atol=1e-9)


def test_coloring_a_smooth_surface_by_value_keeps_where_it_was_moved():
    axes = ThreeDAxes(x_range=[0, 4], y_range=[0, 4], z_range=[0, 4])
    surface = Surface(
        lambda u, v: axes.c2p(u, v, u * v / 4),
        u_range=[0, 4],
        v_range=[0, 4],
        resolution=4,
        checkerboard_colors=False,
    ).shift(2 * UP)
    center = surface.get_center()

    surface.set_fill_by_value(axes, [(RED, 0), (WHITE, 4)], axis=2)

    np.testing.assert_allclose(surface.get_center(), center)


def test_meshes_can_be_in_vector_groups():
    group = VGroup(Square(), Sphere(), Dot3D())

    group.set_color(RED)

    assert group[1].fill_color == RED
    with pytest.raises(TypeError, match="Group instead"):
        VGroup(Square(), Mobject())


def test_surfaces_are_smooth_and_without_outlines_unless_asked():
    smooth = Surface(lambda u, v: [u, v, 0], resolution=4)
    checkered = Surface(
        lambda u, v: [u, v, 0], resolution=4, checkerboard_colors=[RED, BLUE]
    )

    assert not smooth.flat
    assert len(smooth.points) == 5 * 5
    assert smooth.stroke_width == 0
    assert smooth.fill_color == ManimColor(BLUE_D)
    assert checkered.flat
    assert len(checkered.points) == 4 * 4 * 4
    for solid in (Sphere(), Torus(), Cylinder(), Cone(), Dot3D()):
        for part in _meshes(solid):
            assert not part.flat
            assert part.stroke_width == 0


def test_smooth_surface_colours_its_points_by_value():
    from manim.mobject.three_d.three_dimensions import _halfway_around

    axes = ThreeDAxes(x_range=[0, 4], y_range=[0, 4], z_range=[0, 4])
    surface = Surface(
        lambda u, v: axes.c2p(u, v, u * v / 4),
        u_range=[0, 4],
        v_range=[0, 4],
        resolution=4,
    )
    surface.set_fill_by_value(axes, [(RED, 0), (WHITE, 4)], axis=2)

    # Still smooth: one colour per point, blended across the faces.
    assert not surface.flat
    values = axes.point_to_coords(surface.points)[:, 2] / 4
    expected = (1 - values[:, None]) * RED.to_rgb() + values[:, None] * WHITE.to_rgb()
    np.testing.assert_allclose(surface.rgbas[:, :3], expected, atol=1e-9)
    # Each face's outline takes the colour halfway around it.
    values = axes.point_to_coords(_halfway_around(surface))[:, 2] / 4
    expected = (1 - values[:, None]) * RED.to_rgb() + values[:, None] * WHITE.to_rgb()
    np.testing.assert_allclose(surface.stroke_rgbas[:, :3], expected, atol=1e-9)


def test_lines_in_space_are_white_as_lines_are():
    assert {part.fill_color for part in _meshes(Line3D())} == {ManimColor(WHITE)}
    assert {part.fill_color for part in _meshes(Line3D(color=RED))} == {ManimColor(RED)}


def test_partial_surfaces_show_whole_faces_with_their_outlines():
    surface = Surface(
        lambda u, v: [u, v, 0], resolution=4, checkerboard_colors=[RED, BLUE]
    )
    surface.set_fill_by_value(
        ThreeDAxes(x_range=[0, 1], y_range=[0, 1], z_range=[0, 1]), [RED, WHITE], axis=0
    )
    partial = surface.copy().pointwise_become_partial(surface, 0, 0.5)

    assert len(partial.get_faces()) == 8
    np.testing.assert_array_equal(partial.stroke_rgbas, surface.stroke_rgbas[:8])


def test_groups_with_meshes_can_be_rotated_and_recoloured():
    group = VGroup(Square().set_fill([RED, BLUE], opacity=1), Sphere())

    group.rotate(PI / 2).set_color(GREEN)

    np.testing.assert_allclose(group[0].fill.direction, rotate_vector(UL, PI / 2))


SOLIDS_WITH_PARTS = {
    "cone with base": lambda: Cone(show_base=True),
    "cylinder": lambda: Cylinder(),
    "arrow": lambda: Arrow3D(ORIGIN, RIGHT),
}


def _meshes(mob):
    return [m for m in mob.get_family() if isinstance(m, MeshMobject)]


@pytest.mark.parametrize("build", SOLIDS_WITH_PARTS.values(), ids=SOLIDS_WITH_PARTS)
def test_color_and_opacity_reach_every_part_of_a_solid(build):
    solid = build()
    assert len(_meshes(solid)) > 1
    solid.set_color(RED)
    assert {m.fill_color.to_hex() for m in _meshes(solid)} == {RED.to_hex()}
    solid.set_opacity(0.3)
    assert {round(m.fill_opacity, 6) for m in _meshes(solid)} == {0.3}
    solid.set_fill(BLUE, opacity=0.5)
    assert {
        (m.fill_color.to_hex(), round(m.fill_opacity, 6)) for m in _meshes(solid)
    } == {(BLUE.to_hex(), 0.5)}
    solid.set_stroke(WHITE, width=2)
    assert {m.stroke_width for m in _meshes(solid)} == {2}
    solid.fade(0.5)
    assert {round(m.fill_opacity, 6) for m in _meshes(solid)} == {0.25}


def test_family_false_changes_only_the_mesh_itself():
    cone = Cone(show_base=True, fill_color=BLUE)
    base = cone.base_circle
    cone.set_color(RED, family=False)
    cone.set_opacity(0.2, family=False)
    cone.set_stroke(WHITE, width=3, family=False)
    assert cone.fill_color.to_hex() == RED.to_hex()
    assert base.fill_color.to_hex() == BLUE.to_hex()
    assert base.fill_opacity == 1
    assert base.stroke_width == 0
    cone.set_fill(WHITE, opacity=0.7, family=False)
    assert base.fill_color.to_hex() == BLUE.to_hex()
    assert base.fill_opacity == 1


def test_cube_faces_are_lit_meshes_in_a_group():
    cube = Cube(side_length=2, fill_color=RED)

    assert len(cube) == 6
    assert all(isinstance(face, PolygonFace) for face in cube)
    assert all(face.material == PbrMaterial() for face in cube)
    np.testing.assert_allclose(cube.get_center(), 0, atol=1e-12)
    assert cube.width == pytest.approx(2)
    cube.set_color(BLUE)
    assert cube[3].fill_color == BLUE


def test_prisms_are_stretched_cubes():
    prism = Prism(dimensions=[1, 2, 3])

    assert (prism.width, prism.height, prism.depth) == pytest.approx((1, 2, 3))


def test_polyhedron_faces_follow_their_vertices_and_keep_their_style():
    tetrahedron = Tetrahedron(faces_config={"fill_color": RED, "fill_opacity": 0.3})
    face = tetrahedron.faces[0]

    assert isinstance(face, PolygonFace)
    assert face.fill_color == RED
    assert face.fill_opacity == pytest.approx(0.3)
    # Outlined in the face's color, as the Polygon faces were.
    assert face.stroke_color == RED
    assert face.stroke_width == 4

    tetrahedron.graph[0].shift(UP)
    tetrahedron.update()
    first_corner = tetrahedron.faces_list[0].index(0)
    np.testing.assert_allclose(
        face.points[first_corner], tetrahedron.graph[0].get_center()
    )


def test_set_color_with_an_alpha_reaches_the_caps_of_a_cylinder():
    cylinder = Cylinder().set_color(RED, 0.3)
    for cap in (cylinder.base_top, cylinder.base_bottom):
        assert cap.fill_color == RED
        assert cap.fill_opacity == pytest.approx(0.3)


def test_polyhedron_subclasses_can_change_the_default_faces_config():
    class OpaqueTetrahedron(Tetrahedron):
        default_faces_config = {"fill_opacity": 1.0, "stroke_width": 0}

    face = OpaqueTetrahedron().faces[0]
    assert face.fill_opacity == 1.0
    assert face.stroke_width == 0
    assert Tetrahedron().faces[0].fill_opacity == 0.5


def test_surface_subclasses_can_make_their_own_caps():
    class SquareCapped(Cylinder):
        def make_cap(self, radius):
            return Square(2 * radius)

    capped = SquareCapped()
    assert isinstance(capped.base_top, Square)
