"""Semantic camera state shared with rendering backends."""

from __future__ import annotations

__all__ = ["Camera"]

import operator as op
from collections.abc import Iterable
from functools import reduce
from typing import TYPE_CHECKING, Literal, overload

import numpy as np

from manim._config import config
from manim.constants import DOWN, LEFT, RIGHT, UP
from manim.mobject.frame import ScreenRectangle
from manim.mobject.mobject import Mobject, _AnimationBuilder
from manim.mobject.types.vectorized_mobject import VMobject
from manim.utils.color import WHITE, ManimColor, ParsableManimColor
from manim.utils.family import extract_mobject_family_members
from manim.utils.iterables import list_difference_update

if TYPE_CHECKING:
    from manim.typing import (
        FloatRGBA_Array,
        Point3D,
        Point3D_Array,
        Point3DLike,
    )


class Camera:
    """Describe the logical view used by a rendering backend.

    Camera owns an animatable frame, semantic background settings, display ordering,
    and pure point transformations. Raster targets, pixel dimensions, image buffers,
    and backend contexts belong to renderers.
    """

    def __init__(
        self,
        background_image: str | None = None,
        frame_center: Point3DLike | Mobject | None = None,
        frame: Mobject | None = None,
        default_frame_stroke_color: ManimColor = WHITE,
        default_frame_stroke_width: float = 0,
        use_z_index: bool = True,
        frame_height: float | None = None,
        frame_width: float | None = None,
        background_color: ParsableManimColor | None = None,
        background_opacity: float | None = None,
    ) -> None:
        self.background_image = background_image
        self.use_z_index = use_z_index

        if frame is None:
            resolved_height = (
                float(config["frame_height"]) if frame_height is None else frame_height
            )
            resolved_width = (
                float(config["frame_width"]) if frame_width is None else frame_width
            )
            if resolved_height <= 0 or resolved_width <= 0:
                raise ValueError("Camera frame dimensions must be positive.")
            frame = ScreenRectangle(
                aspect_ratio=resolved_width / resolved_height,
                height=resolved_height,
            )
            frame.set_stroke(
                ManimColor(default_frame_stroke_color),
                default_frame_stroke_width,
            )
        else:
            if frame_height is not None:
                if frame_height <= 0:
                    raise ValueError("Camera frame height must be positive.")
                frame.stretch_to_fit_height(frame_height)
            if frame_width is not None:
                if frame_width <= 0:
                    raise ValueError("Camera frame width must be positive.")
                frame.stretch_to_fit_width(frame_width)
        self.frame = frame
        if frame_center is not None:
            self.frame_center = frame_center

        self._background_color = ManimColor(
            config["background_color"]
            if background_color is None
            else background_color,
        )
        self._background_opacity = (
            config["background_opacity"]
            if background_opacity is None
            else background_opacity
        )

    @property
    def background_color(self) -> ManimColor:
        return self._background_color

    @background_color.setter
    def background_color(self, color: ParsableManimColor) -> None:
        self._background_color = ManimColor(color)

    @property
    def background_opacity(self) -> float:
        return self._background_opacity

    @background_opacity.setter
    def background_opacity(self, alpha: float) -> None:
        self._background_opacity = alpha

    @property
    def frame_height(self) -> float:
        """Height of the logical camera frame in Manim units."""
        return self.frame.height

    @frame_height.setter
    def frame_height(self, frame_height: float) -> None:
        self.frame.stretch_to_fit_height(frame_height)

    @property
    def frame_width(self) -> float:
        """Width of the logical camera frame in Manim units."""
        return self.frame.width

    @frame_width.setter
    def frame_width(self, frame_width: float) -> None:
        self.frame.stretch_to_fit_width(frame_width)

    @property
    def frame_center(self) -> Point3D:
        """Center of the logical camera frame."""
        return self.frame.get_center()

    @frame_center.setter
    def frame_center(self, frame_center: Point3DLike | Mobject) -> None:
        self.frame.move_to(frame_center)

    def get_mobjects_to_display(
        self,
        mobjects: Iterable[Mobject],
        include_submobjects: bool = True,
        excluded_mobjects: list[Mobject] | None = None,
    ) -> list[Mobject]:
        """Return the camera-ordered family members visible to the renderer."""
        if include_submobjects:
            mobjects = extract_mobject_family_members(
                mobjects,
                use_z_index=self.use_z_index,
                only_those_with_points=True,
            )
            if excluded_mobjects:
                all_excluded = extract_mobject_family_members(
                    excluded_mobjects,
                    use_z_index=self.use_z_index,
                )
                mobjects = list_difference_update(mobjects, all_excluded)
        return list(mobjects)

    def is_in_frame(self, mobject: Mobject) -> bool:
        """Whether ``mobject`` intersects the logical frame bounds."""
        center = self.frame_center
        height = self.frame_height
        width = self.frame_width
        return not reduce(
            op.or_,
            [
                mobject.get_right()[0] < center[0] - width / 2,
                mobject.get_bottom()[1] > center[1] + height / 2,
                mobject.get_left()[0] > center[0] + width / 2,
                mobject.get_top()[1] < center[1] - height / 2,
            ],
        )

    def get_mobjects_indicating_movement(self) -> list[Mobject]:
        """Camera controls whose animation changes every projected pixel."""
        return [self.frame]

    @overload
    def auto_zoom(
        self,
        mobjects: Iterable[Mobject],
        margin: float = 0,
        only_mobjects_in_frame: bool = False,
        animate: Literal[False] = False,
    ) -> Mobject: ...

    @overload
    def auto_zoom(
        self,
        mobjects: Iterable[Mobject],
        margin: float = 0,
        only_mobjects_in_frame: bool = False,
        animate: Literal[True] = True,
    ) -> _AnimationBuilder: ...

    def auto_zoom(
        self,
        mobjects: Iterable[Mobject],
        margin: float = 0,
        only_mobjects_in_frame: bool = False,
        animate: bool = True,
    ) -> _AnimationBuilder | Mobject:
        """Move and resize the frame to contain the supplied 2D mobjects."""
        (
            left,
            right,
            top,
            bottom,
        ) = self._get_bounding_box(mobjects, only_mobjects_in_frame)
        x = (left + right) / 2
        y = (top + bottom) / 2
        new_width = abs(left - right)
        new_height = abs(top - bottom)
        target = self.frame.animate if animate else self.frame
        if new_width / self.frame.width > new_height / self.frame.height:
            return target.set_x(x).set_y(y).set(width=new_width + margin)
        return target.set_x(x).set_y(y).set(height=new_height + margin)

    def _get_bounding_box(
        self,
        mobjects: Iterable[Mobject],
        only_mobjects_in_frame: bool,
    ) -> tuple[float, float, float, float]:
        bounds: tuple[float, float, float, float] | None = None
        for mobject in mobjects:
            if mobject is self.frame or (
                only_mobjects_in_frame and not self.is_in_frame(mobject)
            ):
                continue
            mobject_bounds = (
                float(mobject.get_critical_point(LEFT)[0]),
                float(mobject.get_critical_point(RIGHT)[0]),
                float(mobject.get_critical_point(UP)[1]),
                float(mobject.get_critical_point(DOWN)[1]),
            )
            if bounds is None:
                bounds = mobject_bounds
            else:
                bounds = (
                    min(bounds[0], mobject_bounds[0]),
                    max(bounds[1], mobject_bounds[1]),
                    max(bounds[2], mobject_bounds[2]),
                    min(bounds[3], mobject_bounds[3]),
                )
        if bounds is None:
            raise ValueError(
                "Could not determine the bounding box of the mobjects given to "
                "Camera.auto_zoom().",
            )
        return bounds

    def _prepare_for_render(self) -> None:
        """Refresh derived semantic view state before a renderer borrows it."""

    def get_view_transform_center(self) -> Point3D:
        """Return the center applied by the renderer's 2D view transform."""
        return self.frame_center

    def get_stroke_rgbas(
        self,
        vmobject: VMobject,
        background: bool = False,
    ) -> FloatRGBA_Array:
        """Return stroke colors after camera-specific semantic shading."""
        return vmobject.get_stroke_rgbas(background)

    def get_fill_rgbas(self, vmobject: VMobject) -> FloatRGBA_Array:
        """Return fill colors after camera-specific semantic shading."""
        return vmobject.get_fill_rgbas()

    def transform_points_pre_display(
        self,
        mobject: Mobject,
        points: Point3D_Array,
    ) -> Point3D_Array:
        """Apply camera-specific pure projection before display."""
        if not np.all(np.isfinite(points)):
            return np.zeros((1, 3))
        return points
