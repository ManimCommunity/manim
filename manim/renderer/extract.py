"""Snapshots of mobjects for renderers (see :mod:`manim.renderer.snapshot`).

This is the only code that reads mobjects to make what renderers draw. A mobject is
drawn from its own fields: nothing in its snapshot comes from its family, so a parent's
snapshot does not change when its children move. Gradient ends and the depth
reference come from the bounding box of the mobject's own anchors.
"""

from __future__ import annotations

import os
import weakref
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from manim.appearance.material import UnlitMaterial
from manim.mobject.mobject import Mobject
from manim.mobject.three_d.mesh import MeshMobject
from manim.mobject.types.image_mobject import (
    AbstractImageMobject,
    ImageMobjectFromCamera,
)
from manim.mobject.types.point_cloud_mobject import PMobject
from manim.mobject.types.vectorized_mobject import VMobject
from manim.renderer.snapshot import (
    FacingCamera,
    FixedInFrame,
    InScene,
    LegacyShaded,
    Payload,
    Placement,
    RCameraView,
    RImage,
    RLinearGradient,
    RMesh,
    RNode,
    RPaint,
    RPath,
    RPoints,
    RSolidColor,
    RStroke,
)
from manim.utils._path_points import end_corner, start_corner
from manim.utils.family import extract_mobject_family_members
from manim.utils.iterables import list_difference_update

if TYPE_CHECKING:
    from manim.typing import FloatRGBA_Array, Point3D

__all__ = ["Extractor", "extract", "placement_of", "snapshot"]

_UNLIT = UnlitMaterial()
_LEGACY_SHADED = LegacyShaded()
_FIXED_IN_FRAME = FixedInFrame()
_IN_SCENE = InScene()


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

    Every snapshot is taken anew; an :class:`Extractor` reuses those of unchanged
    mobjects.

    Raises
    ------
    TypeError
        If a mobject is of a type no renderer can draw.
    """
    nodes: list[RNode] = []
    for mobject in _members(
        mobjects, include_submobjects, use_z_index, excluded_mobjects
    ):
        node = snapshot(mobject)
        if node is not None:
            nodes.append(node)
    return nodes


#: Compare every reused snapshot with a fresh one (set in Manim's tests).
CHECK_SNAPSHOTS = os.environ.get("MANIM_CHECK_SNAPSHOTS", "") not in ("", "0")


class Extractor:
    """Takes render snapshots like :func:`extract`, reusing the snapshot of a mobject
    while its fields keep their versions.

    This relies on one rule: a snapshot is a function of the mobject's own fields.
    Mobjects whose snapshot has no key (see :attr:`.RNode.key`: classes with
    ``_drawn_from_fields = False``, camera views, paths masked by an image object)
    and mobjects with a fixed orientation, whose center comes from a function, are
    taken anew every time. With :data:`CHECK_SNAPSHOTS` (``MANIM_CHECK_SNAPSHOTS=1``,
    as in Manim's tests), every reused snapshot is compared with a fresh one, so a
    class that breaks the rule fails instead of showing stale pixels.

    Whoever draws frames owns an extractor; it holds the mobjects it saw weakly.
    """

    def __init__(self) -> None:
        self._snapshots: weakref.WeakKeyDictionary[Mobject, tuple[int, RNode]] = (
            weakref.WeakKeyDictionary()
        )

    def extract(
        self,
        mobjects: Iterable[Mobject],
        *,
        include_submobjects: bool = True,
        use_z_index: bool = True,
        excluded_mobjects: Iterable[Mobject] = (),
    ) -> list[RNode]:
        """Snapshots as :func:`extract` returns them."""
        nodes: list[RNode] = []
        snapshots = self._snapshots
        for mobject in _members(
            mobjects, include_submobjects, use_z_index, excluded_mobjects
        ):
            revision = _revision(mobject.__dict__.get("_field_versions", {}))
            cached = snapshots.get(mobject)
            node: RNode | None
            if cached is not None and cached[0] == revision:
                node = cached[1]
                if CHECK_SNAPSHOTS:
                    _check_unchanged(mobject, node)
            else:
                node = snapshot(mobject)
                # The orientation center comes from a function, called every time.
                if (
                    node is not None
                    and node.key is not None
                    and mobject.fixed_orientation is None
                ):
                    snapshots[mobject] = (revision, node)
                else:
                    snapshots.pop(mobject, None)
            if node is not None:
                nodes.append(node)
        return nodes


def _members(
    mobjects: Iterable[Mobject],
    include_submobjects: bool,
    use_z_index: bool,
    excluded_mobjects: Iterable[Mobject],
) -> list[Mobject]:
    if not include_submobjects:
        return list(mobjects)
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
    return members


def _revision(versions: Mapping[str, int]) -> int:
    """Changes whenever a field of the mobject is assigned: the largest version.

    Versions come from one counter that only goes up (see
    :mod:`manim.utils.versioned_fields`), so every assignment, also the first one of
    a field, gives the mobject a larger largest version.
    """
    return max(versions.values(), default=0)


def _check_unchanged(mobject: Mobject, node: RNode) -> None:
    if not _same(node, snapshot(mobject)):
        raise AssertionError(
            f"The render snapshot of {mobject!r} changed although none of its fields "
            "did. Draw it from its own fields, or set "
            f"{type(mobject).__name__}._drawn_from_fields = False."
        )


def _same(a: Any, b: Any) -> bool:
    if a is b:
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return bool(np.array_equal(a, b))
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return bool(a == b)


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


def placement_of(mobject: Mobject) -> Placement:
    """How ``mobject`` is placed in the view (see :attr:`.RNode.placement`)."""
    if mobject.is_fixed_in_frame:
        return _FIXED_IN_FRAME
    center_func = mobject._get_fixed_orientation_anchor
    if center_func is not None:
        return FacingCamera(np.asarray(center_func(), dtype=np.float64))
    return _IN_SCENE


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
    placement: Placement,
) -> Any:
    """The node's key (see :attr:`.RNode.key`): its uid, the largest version of its
    fields (see :func:`_revision`), and the point a fixed orientation follows (not a
    field value); ``None`` for mobjects that can change without a field changing.
    """
    if (
        not mobject._drawn_from_fields
        or isinstance(payload, RCameraView)
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
    return (mobject.uid, _revision(versions), center)


def _payload(mobject: Mobject) -> tuple[Payload | None, dict[str, tuple[str, ...]]]:
    """The payload of ``mobject`` and the fields each of its slots is made from."""
    if isinstance(mobject, VMobject):
        return _path(mobject), _PATH_SLOTS
    if isinstance(mobject, MeshMobject):
        return RMesh(
            mobject.points,
            mobject.triangles,
            mobject.rgbas,
            mobject.uvs,
            mobject.texture,
            mobject.material,
            mobject.depth_test,
            mobject.faces,
            mobject.stroke_rgbas,
            mobject.stroke_width,
        ), _MESH_SLOTS
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


_MESH_SLOTS: dict[str, tuple[str, ...]] = {
    "geometry": ("points", "triangles", "faces"),
    "rgbas": ("rgbas",),
    "uvs": ("uvs",),
    "texture": ("texture",),
    "material": ("material",),
    "stroke": ("stroke_rgbas", "stroke_width"),
}

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
            return RSolidColor(rgbas[0])
        return RLinearGradient(ends[0], ends[1], rgbas)

    background_width = vmobject.background_stroke_width
    return RPath(
        points=points,
        fill=paint(fill),
        stroke=RStroke(
            paint(stroke),
            vmobject.stroke_width,
            vmobject.joint_type,
            vmobject.cap_style,
        ),
        background_stroke=(
            RStroke(
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
