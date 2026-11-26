from __future__ import annotations

import numpy as np
import pytest

from manim import (
    BLUE,
    DOWN,
    GREEN,
    LEFT,
    RED,
    RIGHT,
    UP,
    Arrow,
    Circle,
    ImageMobject,
    Line,
    Mobject,
    PMobject,
    Square,
    VGroup,
    VMobject,
)
from manim.appearance.material import PbrMaterial, UnlitMaterial
from manim.mobject.types.image_mobject import AbstractImageMobject
from manim.renderer.extract import extract, snapshot
from manim.renderer.snapshot import (
    FacingCamera,
    FixedInFrame,
    InScene,
    LegacyShaded,
    RImage,
    RLinearGradient,
    RPath,
    RPoints,
    RSolidColor,
)


def test_extract_skips_excluded_families_and_mobjects_without_drawing():
    square, circle = Square(), Circle()
    cloud = PMobject().add_points([[0, 0, 0]])
    image = ImageMobject(np.zeros((2, 2, 4), dtype=np.uint8))
    excluded = VGroup(Square(), Square())
    plain = Mobject()
    plain.points = [[0, 0, 0]]

    nodes = extract(
        [VGroup(square, circle), cloud, image, excluded, plain],
        excluded_mobjects=[excluded],
    )

    assert [node.uid for node in nodes] == [
        m.uid for m in (square, circle, cloud, image)
    ]
    assert [type(node.payload) for node in nodes] == [RPath, RPath, RPoints, RImage]


def test_snapshot_shares_field_values():
    square = Square()
    node = snapshot(square)
    assert node.payload.points is square.points
    assert isinstance(node.payload.fill, RSolidColor)
    np.testing.assert_array_equal(node.payload.fill.rgba, square.fill_rgbas[0])
    assert node.payload.material == UnlitMaterial()
    assert node.payload.depth_reference is None


def test_key_changes_with_any_field_and_only_then():
    square = Square()
    key = snapshot(square).key
    assert snapshot(square).key == key

    square.set_fill(RED, opacity=1)
    assert snapshot(square).key != key
    key = snapshot(square).key

    square.z_index = 3
    assert snapshot(square).key != key


def test_the_key_is_the_newest_assignment_to_any_field():
    square, other = Square(), Square()
    key = snapshot(square).key

    other.set_fill(RED)  # another mobject
    assert snapshot(square).key == key

    square.stroke_width = square.stroke_width + 1
    assert snapshot(square).key != key


@pytest.mark.parametrize("fixed_orientation", [False, True])
def test_fixed_in_frame_takes_precedence_over_a_fixed_orientation(fixed_orientation):
    square = Square()
    if fixed_orientation:
        square.fix_orientation()
    square.fix_in_frame()
    assert snapshot(square).placement == FixedInFrame()
    assert snapshot(square).fixed_in_frame

    square.unfix_from_frame()
    placement = snapshot(square).placement
    if fixed_orientation:
        assert isinstance(placement, FacingCamera)
        np.testing.assert_array_equal(placement.center, square.get_center())
    else:
        assert placement == InScene()
    assert not snapshot(square).fixed_in_frame


def test_copies_have_other_keys():
    square = Square()
    assert snapshot(square.copy()).key != snapshot(square).key


def test_slot_versions_change_with_their_inputs():
    square = Square()
    before = snapshot(square).versions

    square.set_fill(RED, opacity=1)
    after = snapshot(square).versions
    assert {s for s in before if before[s] != after[s]} == {"fill"}

    square.tolerance_for_point_equality = 1e-3
    tolerance = snapshot(square).versions
    assert tolerance["geometry"] != after["geometry"]


def test_a_moved_gradient_gets_new_paint_versions():
    square = Square().set_fill([RED, BLUE], opacity=1)
    before = snapshot(square)

    square.shift(RIGHT)
    after = snapshot(square)

    assert after.versions["fill"] != before.versions["fill"]
    np.testing.assert_allclose(
        after.payload.fill.start, before.payload.fill.start + RIGHT
    )


def test_paints_and_materials():
    square = Square().set_fill([RED, BLUE], opacity=1).set_stroke(GREEN, width=3)
    path = snapshot(square).payload

    assert isinstance(path.fill, RLinearGradient)
    np.testing.assert_allclose(path.fill.rgbas, square.fill_rgbas)
    assert isinstance(path.stroke.paint, RSolidColor)
    assert path.stroke.width == 3
    assert path.background_stroke is None

    square.set_shade_in_3d(True)
    shaded = snapshot(square).payload
    assert isinstance(shaded.material, LegacyShaded)
    np.testing.assert_allclose(shaded.depth_reference, square.get_center())


def test_materials_are_value_types():
    assert PbrMaterial() == PbrMaterial(roughness=0.5, metallic=0.0, emissive=(0, 0, 0))
    assert PbrMaterial(emissive=[1, 0, 0]).emissive == (1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="roughness"):
        PbrMaterial(roughness=2)
    with pytest.raises(ValueError, match="emissive"):
        PbrMaterial(emissive=(-1, 0, 0))


# A mobject is drawn from its own geometry: its children do not change its snapshot.


def _parent_with_child():
    parent = Square().set_fill([RED, BLUE], opacity=1).set_shade_in_3d(True)
    child = Circle().shift(3 * RIGHT)
    parent.add(child)
    return parent, child


@pytest.mark.parametrize(
    "change",
    [
        lambda parent, child: child.shift(2 * UP),
        lambda parent, child: parent.add(Circle().shift(4 * DOWN)),
        lambda parent, child: parent.remove(child),
    ],
    ids=["move", "add", "remove"],
)
def test_children_do_not_change_the_parent_snapshot(change):
    parent, child = _parent_with_child()
    before = snapshot(parent)

    change(parent, child)
    after = snapshot(parent)

    assert after.key == before.key
    assert after.payload.fill.start.tolist() == before.payload.fill.start.tolist()
    assert after.payload.fill.end.tolist() == before.payload.fill.end.tolist()
    assert (
        after.payload.depth_reference.tolist()
        == before.payload.depth_reference.tolist()
    )


def test_gradients_span_the_mobjects_own_points():
    arrow = Arrow(2 * LEFT, 2 * RIGHT).set_stroke([RED, BLUE])
    path = snapshot(arrow).payload
    anchors = np.concatenate([arrow.points[::4], arrow.points[3::4]])
    center = (anchors.min(axis=0) + anchors.max(axis=0)) / 2
    np.testing.assert_allclose(
        (path.stroke.paint.start + path.stroke.paint.end) / 2, center
    )


# Nodes that can change without a field change have no key.


def test_image_subclasses_that_compute_pixels_have_no_key():
    class Noise(AbstractImageMobject):
        def __init__(self):
            super().__init__(scale_to_resolution=0)

        def get_pixel_array(self):
            return np.zeros((2, 2, 4), dtype=np.uint8)

    assert snapshot(Noise()).key is None
    assert snapshot(ImageMobject(np.zeros((2, 2, 4), dtype=np.uint8))).key is not None


def test_masked_paths_keep_their_paint():
    stroke = Line(LEFT, RIGHT).set_stroke(RED, width=4, opacity=0.2)
    stroke.color_using_background_image("some_image.png")
    node = snapshot(stroke)
    assert node.payload.mask == "some_image.png"
    # The mask multiplies the path's own paint, so the paint stays.
    assert isinstance(node.payload.stroke.paint, RSolidColor)


def test_point_clouds_have_no_key_yet():
    cloud = PMobject().add_points([[0, 0, 0]])
    assert snapshot(cloud).key is None


def test_empty_paths_are_drawn_without_errors():
    assert snapshot(VMobject()) is not None


def test_the_key_follows_a_fixed_orientation_center():
    anchor = [np.zeros(3)]
    label = Square().fix_orientation(lambda: anchor[0])
    key = snapshot(label).key

    anchor[0] = np.array([1.0, 2.0, 3.0])

    assert snapshot(label).key != key


@pytest.mark.parametrize("camera_view", [False, True])
def test_resampling_algorithms_given_as_numbers_are_stored_as_the_enum(camera_view):
    from PIL.Image import Resampling

    from manim import Camera
    from manim.mobject.types.image_mobject import ImageMobjectFromCamera

    if camera_view:
        image = ImageMobjectFromCamera(Camera())
    else:
        image = ImageMobject(np.zeros((2, 2, 4), dtype=np.uint8))
    image.set_resampling_algorithm(0)
    assert type(image.resampling_algorithm) is Resampling
    assert snapshot(image).payload.resampling_algorithm is Resampling.NEAREST
