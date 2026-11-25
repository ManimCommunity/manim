"""Snapshots of mobjects for renderers (see :mod:`manim.renderer.snapshot`).

This is the only code that reads mobjects to make what renderers draw. A mobject is
drawn from its own fields: nothing in its snapshot comes from its family, so a parent's
snapshot does not change when its children move. Gradient ends and the depth
reference come from the bounding box of the mobject's own anchors.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np

from manim.material import UnlitMaterial
from manim.mobject.mobject import Mobject
from manim.mobject.types.image_mobject import (
    AbstractImageMobject,
    ImageMobjectFromCamera,
)
from manim.mobject.types.point_cloud_mobject import PMobject
from manim.mobject.types.vectorized_mobject import VMobject
from manim.renderer.snapshot import (
    FacingCamera,
    FixedInFrame,
    LegacyShaded,
    Payload,
    Placement,
    RCameraView,
    RImage,
    RLinearGradient,
    RNode,
    RPaint,
    RPath,
    RPoints,
    RSolid,
    Stroke,
)
from manim.utils._path_points import end_corner, start_corner
from manim.utils.family import extract_mobject_family_members
from manim.utils.iterables import list_difference_update

if TYPE_CHECKING:
    from manim.typing import FloatRGBA_Array, Point3D

__all__ = ["extract", "placement_of", "snapshot"]

_UNLIT = UnlitMaterial()
_LEGACY_SHADED = LegacyShaded()
_FIXED_IN_FRAME = FixedInFrame()


def extract(
    mobjects: Iterable[Mobject],
    *,
    include_submobjects: bool = True,
    use_z_index: bool = True,
    excluded_mobjects: Iterable[Mobject] = (),
) -> list[RNode]:
    """Return render snapshots for the families of ``mobjects`` in drawing order.

    The order is family order, stably sorted by ``z_index`` if ``use_z_index``. A
    mobject that occurs more than once (in several families) is drawn once, at its
    last place. Families of ``excluded_mobjects`` and mobjects without points are
    left out. With ``include_submobjects=False``, ``mobjects`` are taken as given.
    Plain :class:`~.Mobject` instances are always left out; they have nothing to draw.

    Raises
    ------
    TypeError
        If a mobject is of a type no renderer can draw.
    """
    if not include_submobjects:
        members = list(mobjects)
    else:
        members = extract_mobject_family_members(
            mobjects,
            use_z_index=use_z_index,
            only_those_with_points=True,
        )
        excluded = list(excluded_mobjects)
        if excluded:
            members = list_difference_update(
                members,
                extract_mobject_family_members(excluded, use_z_index=use_z_index),
            )
    nodes: list[RNode] = []
    for mobject in members:
        node = snapshot(mobject)
        if node is not None:
            nodes.append(node)
    return nodes


def snapshot(mobject: Mobject) -> RNode | None:
    """Return the render snapshot of ``mobject`` alone (not of its family), or
    ``None`` if it has nothing to draw.

    Raises
    ------
    TypeError
        If the mobject is of a type no renderer can draw.
    """
    payload, slots = _payload(mobject)
    if payload is None:
        return None
    versions: dict[str, int] = mobject.__dict__.get("_field_versions", {})
    placement = placement_of(mobject)
    return RNode(
        uid=mobject.uid,
        key=_key(mobject, payload, versions, placement),
        versions=_slot_versions(versions, slots),
        payload=payload,
        placement=placement,
    )


def placement_of(mobject: Mobject) -> Placement | None:
    """How ``mobject`` is placed in the view (see :attr:`.RNode.placement`)."""
    if mobject.fixed_in_frame:
        return _FIXED_IN_FRAME
    center_func = mobject.fixed_orientation
    if center_func is not None:
        return FacingCamera(np.asarray(center_func(), dtype=np.float64))
    return None


def _slot_versions(
    versions: dict[str, int], slots: dict[str, tuple[str, ...]]
) -> dict[str, int]:
    """The version of each slot: the largest version of the fields it is made of."""
    get = versions.get
    result: dict[str, int] = {}
    for slot, names in slots.items():
        latest = 0
        for name in names:
            version = get(name, 0)
            if version > latest:
                latest = version
        result[slot] = latest
    return result


def _key(
    mobject: Mobject,
    payload: Payload,
    versions: dict[str, int],
    placement: Placement | None,
) -> Any:
    """The node's key (see :attr:`.RNode.key`): its uid, the largest version of its
    fields, and the point a fixed orientation follows (not a field value); ``None``
    for mobjects that can change without a field changing.

    Versions come from one counter (see :mod:`manim.utils.versioned_fields`), so the
    largest one changes whenever any field of the mobject is assigned.
    """
    if (
        not mobject._drawn_from_fields
        # Point clouds: their colors are not a field yet.
        or isinstance(payload, (RCameraView, RPoints))
        # An image object can change without being assigned again.
        or (
            isinstance(payload, RPath)
            and not isinstance(payload.mask, (str, type(None)))
        )
    ):
        return None
    center = (
        tuple(placement.center.tolist())
        if isinstance(placement, FacingCamera)
        else None
    )
    return (mobject.uid, max(versions.values(), default=0), center)


def _payload(mobject: Mobject) -> tuple[Payload | None, dict[str, tuple[str, ...]]]:
    """The payload of ``mobject`` and the fields each of its slots is made from."""
    if isinstance(mobject, VMobject):
        return _path(mobject), _PATH_SLOTS
    if isinstance(mobject, PMobject):
        return RPoints(mobject.points, mobject.rgbas, mobject.stroke_width), {
            "geometry": ("points",),
            "colors": ("rgbas", "stroke_width"),
        }
    if isinstance(mobject, AbstractImageMobject):
        return RImage(
            mobject.get_pixel_array(), mobject.points, mobject.resampling_algorithm
        ), {
            "geometry": ("points",),
            "pixels": ("pixel_array", "resampling_algorithm"),
        }
    if isinstance(mobject, ImageMobjectFromCamera):
        return RCameraView(
            mobject.points, mobject.resampling_algorithm, id(mobject.camera)
        ), {"geometry": ("points",)}
    if isinstance(mobject, Mobject):
        return None, {}
    raise TypeError(
        f"Displaying an object of class {type(mobject).__name__} is not supported"
    )


# The fields each slot of a path is made from. A gradient's ends come from the points.
_PATH_SLOTS: dict[str, tuple[str, ...]] = {
    "geometry": ("points", "tolerance_for_point_equality"),
    "fill": ("fill_rgbas", "sheen_direction", "points", "shade_in_3d"),
    "stroke": (
        "stroke_rgbas",
        "stroke_width",
        "joint_type",
        "cap_style",
        "sheen_direction",
        "points",
        "shade_in_3d",
    ),
    "background_stroke": (
        "background_stroke_rgbas",
        "background_stroke_width",
        "joint_type",
        "cap_style",
        "sheen_direction",
        "points",
        "shade_in_3d",
    ),
    "material": ("shade_in_3d",),
    "depth_reference": ("shade_in_3d", "points"),
    "mask": ("background_image",),
}


def _path(vmobject: VMobject) -> RPath:
    attributes: dict[str, Any] = vmobject.__dict__
    points = vmobject.points
    shade_in_3d = vmobject.shade_in_3d
    fill = attributes.get("fill_rgbas", _NO_COLOR)
    stroke = attributes.get("stroke_rgbas", _NO_COLOR)
    background = attributes.get("background_stroke_rgbas", _NO_COLOR)
    ends = None
    if max(len(fill), len(stroke), len(background)) > 1:
        ends = _gradient_ends(vmobject, points, shade_in_3d)

    def paint(rgbas: FloatRGBA_Array) -> RPaint:
        if ends is None or len(rgbas) == 1:
            return RSolid(rgbas[0])
        return RLinearGradient(ends[0], ends[1], rgbas)

    background_width = vmobject.get_stroke_width(background=True)
    return RPath(
        points=points,
        fill=paint(fill),
        stroke=Stroke(
            paint(stroke),
            vmobject.get_stroke_width(),
            vmobject.joint_type,
            vmobject.cap_style,
        ),
        background_stroke=(
            Stroke(
                paint(background),
                background_width,
                vmobject.joint_type,
                vmobject.cap_style,
            )
            if background_width > 0
            else None
        ),
        material=_LEGACY_SHADED if shade_in_3d else _UNLIT,
        depth_reference=_own_center(points) if shade_in_3d else None,
        tolerance=vmobject.tolerance_for_point_equality,
        mask=vmobject.background_image,
    )


def _anchor_box(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """The bounding box of the anchors of a path's own points (not its family's)."""
    if len(points) == 0:
        return None
    # As VMobject.get_anchors: the first and last point of every curve.
    anchors = np.concatenate([points[::4], points[3::4]]) if len(points) > 1 else points
    return anchors.min(axis=0), anchors.max(axis=0)


def _own_center(points: np.ndarray) -> Point3D:
    box = _anchor_box(points)
    if box is None:
        return np.zeros(3)
    return (box[0] + box[1]) / 2


def _gradient_ends(
    vmobject: VMobject, points: np.ndarray, shade_in_3d: bool
) -> tuple[Point3D, Point3D]:
    """Where a path's gradient starts and ends, from the path's own points."""
    if shade_in_3d:
        # As the 3D shading always did: between the path's first and last corner.
        return start_corner(points), end_corner(points)
    box = _anchor_box(points)
    if box is None:
        return np.zeros(3), np.zeros(3)
    low, high = box
    center = (low + high) / 2
    # Along the sheen direction, to the edges of the box (as
    # get_gradient_start_and_end_points does with the family's box).
    offset = (high - low) / 2 * np.asarray(vmobject.sheen_direction, dtype=float)
    return center - offset, center + offset


_NO_COLOR = np.zeros((1, 4))
_NO_COLOR.flags.writeable = False
