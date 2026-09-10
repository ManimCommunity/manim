"""Private drawing helpers for :class:`~manim.renderer.cairo.CairoRenderer`."""

from __future__ import annotations

import itertools as it
import operator as op
from collections.abc import Callable, Iterable
from functools import reduce
from typing import TYPE_CHECKING, Any

import cairo
import numpy as np
from PIL import Image

from manim.constants import CapStyleType, LineJointType
from manim.mobject.mobject import Mobject
from manim.mobject.types.image_mobject import (
    AbstractImageMobject,
    ImageMobjectFromCamera,
)
from manim.mobject.types.point_cloud_mobject import PMobject
from manim.mobject.types.vectorized_mobject import VMobject
from manim.utils.space_ops import cross2d

from .target import _CairoRenderTarget

if TYPE_CHECKING:
    from manim.typing import (
        FloatRGBA_Array,
        FloatRGBALike_Array,
        Point3D_Array,
        RGBAPixelArray,
    )

    from .camera import Camera

_LINE_JOIN_MAP = {
    LineJointType.AUTO: None,
    LineJointType.ROUND: cairo.LineJoin.ROUND,
    LineJointType.BEVEL: cairo.LineJoin.BEVEL,
    LineJointType.MITER: cairo.LineJoin.MITER,
}

_CAP_STYLE_MAP = {
    CapStyleType.AUTO: None,
    CapStyleType.ROUND: cairo.LineCap.ROUND,
    CapStyleType.BUTT: cairo.LineCap.BUTT,
    CapStyleType.SQUARE: cairo.LineCap.SQUARE,
}

_ImageResolver = Callable[[ImageMobjectFromCamera], np.ndarray | None]


class _CairoDrawingContext:
    """Draw mobjects into one renderer-owned target using one semantic camera."""

    def __init__(
        self,
        *,
        camera: Camera,
        target: _CairoRenderTarget,
        image_resolver: _ImageResolver,
    ) -> None:
        self.camera = camera
        self.target = target
        self.image_resolver = image_resolver

    def draw(
        self,
        mobjects: Iterable[Mobject],
        *,
        include_submobjects: bool = True,
        excluded_mobjects: list[Mobject] | None = None,
    ) -> None:
        self.target._ensure_open()
        display_funcs: dict[type[Mobject], Callable[[list[Any]], None]] = {
            VMobject: self._display_vectorized_mobjects,
            PMobject: self._display_point_cloud_mobjects,
            ImageMobjectFromCamera: self._display_image_mobjects,
            AbstractImageMobject: self._display_image_mobjects,
            Mobject: lambda batch: None,
        }

        def type_or_raise(mobject: Mobject) -> type[Mobject]:
            for mobject_type in display_funcs:
                if isinstance(mobject, mobject_type):
                    return mobject_type
            raise TypeError(
                f"Displaying an object of class {type(mobject).__name__} is not supported",
            )

        self.camera._prepare_for_render()
        to_display = self.camera.get_mobjects_to_display(
            mobjects,
            include_submobjects=include_submobjects,
            excluded_mobjects=excluded_mobjects,
        )
        for group_type, group in it.groupby(to_display, type_or_raise):
            display_funcs[group_type](list(group))

    def _display_vectorized_mobjects(self, vmobjects: list[VMobject]) -> None:
        if not vmobjects:
            return
        for image, batch in it.groupby(
            vmobjects,
            lambda vmobject: vmobject.get_background_image(),
        ):
            if image:
                self._display_background_colored_vmobjects(list(batch))
            else:
                self._display_non_background_colored_vmobjects(batch)

    def _display_non_background_colored_vmobjects(
        self,
        vmobjects: Iterable[VMobject],
    ) -> None:
        context = self.target.get_context(self.camera)
        for vmobject in vmobjects:
            self._display_vectorized(vmobject, context)

    def _display_vectorized(
        self,
        vmobject: VMobject,
        context: cairo.Context,
    ) -> None:
        self._set_cairo_context_path(context, vmobject)
        self._apply_stroke(context, vmobject, background=True)
        self._apply_fill(context, vmobject)
        self._apply_stroke(context, vmobject)

    def _set_cairo_context_path(
        self,
        context: cairo.Context,
        vmobject: VMobject,
    ) -> None:
        points = self.camera.transform_points_pre_display(vmobject, vmobject.points)
        if len(points) == 0:
            return

        nppcc = vmobject.n_points_per_cubic_curve
        split_indices = vmobject.get_subpath_split_indices_from_points(points, n_dims=2)
        if len(split_indices) == 0:
            return

        points_xy = points[:, :2].ravel()
        context.new_path()
        move_to = context.move_to
        curve_to = context.curve_to
        new_sub_path = context.new_sub_path
        close_path = context.close_path

        for start_index, end_index in split_indices:
            start_index = int(start_index)
            end_index = int(end_index)
            if end_index - start_index < nppcc:
                continue

            new_sub_path()
            base = start_index * 2
            move_to(points_xy[base], points_xy[base + 1])
            for index in range(
                start_index,
                end_index - nppcc + 1,
                nppcc,
            ):
                handle = (index + 1) * 2
                curve_to(
                    points_xy[handle],
                    points_xy[handle + 1],
                    points_xy[handle + 2],
                    points_xy[handle + 3],
                    points_xy[handle + 4],
                    points_xy[handle + 5],
                )
            if vmobject.consider_points_equals_2d(
                points[start_index],
                points[end_index - 1],
            ):
                close_path()

    def _set_cairo_context_color(
        self,
        context: cairo.Context,
        rgbas: FloatRGBALike_Array,
        vmobject: VMobject,
    ) -> None:
        if len(rgbas) == 1:
            context.set_source_rgba(*rgbas[0][2::-1], rgbas[0][3])
            return

        points = vmobject.get_gradient_start_and_end_points()
        points = self.camera.transform_points_pre_display(vmobject, points)
        pattern = cairo.LinearGradient(
            *it.chain(*(point[:2] for point in points)),
        )
        for rgba, offset in zip(
            rgbas,
            np.linspace(0, 1, len(rgbas)),
            strict=True,
        ):
            pattern.add_color_stop_rgba(offset, *rgba[2::-1], rgba[3])
        context.set_source(pattern)

    def _apply_fill(self, context: cairo.Context, vmobject: VMobject) -> None:
        self._set_cairo_context_color(
            context,
            self.camera.get_fill_rgbas(vmobject),
            vmobject,
        )
        context.fill_preserve()

    def _apply_stroke(
        self,
        context: cairo.Context,
        vmobject: VMobject,
        background: bool = False,
    ) -> None:
        width = vmobject.get_stroke_width(background)
        if width == 0:
            return
        self._set_cairo_context_color(
            context,
            self.camera.get_stroke_rgbas(vmobject, background=background),
            vmobject,
        )
        context.set_line_width(
            width * self.target.settings.cairo_line_width_multiple,
        )
        if vmobject.joint_type != LineJointType.AUTO:
            context.set_line_join(_LINE_JOIN_MAP[vmobject.joint_type])
        if vmobject.cap_style != CapStyleType.AUTO:
            context.set_line_cap(_CAP_STYLE_MAP[vmobject.cap_style])
        context.stroke_preserve()

    def _display_background_colored_vmobjects(
        self,
        vmobjects: list[VMobject],
    ) -> None:
        scratch = self.target.get_scratch_target()
        scratch_context = _CairoDrawingContext(
            camera=self.camera,
            target=scratch,
            image_resolver=self.image_resolver,
        )
        current: RGBAPixelArray | None = None
        for image, batch in it.groupby(
            vmobjects,
            lambda vmobject: vmobject.get_background_image(),
        ):
            scratch.clear()
            scratch_context._display_non_background_colored_vmobjects(batch)
            background = self.target.get_background_image(image)
            colored = np.asarray(
                background * scratch.pixels.astype(float) / 255,
                dtype=np.uint8,
            )
            current = colored if current is None else np.maximum(current, colored)
        if current is not None:
            self._overlay_rgba_array(self.target.pixels, current)

    def _display_point_cloud_mobjects(self, pmobjects: list[PMobject]) -> None:
        for pmobject in pmobjects:
            self._display_point_cloud(
                pmobject,
                pmobject.points,
                pmobject.rgbas,
                self._adjusted_thickness(pmobject.stroke_width),
            )

    def _display_point_cloud(
        self,
        pmobject: PMobject,
        points: Point3D_Array,
        rgbas: FloatRGBA_Array,
        thickness: float,
    ) -> None:
        if len(points) == 0:
            return
        pixel_coords = self._points_to_pixel_coords(pmobject, points)
        pixel_coords = self._thickened_coordinates(pixel_coords, thickness)
        pixel_array = self.target.pixels
        rgba_len = pixel_array.shape[2]

        int_rgbas = (255 * rgbas).astype(np.uint8)
        target_len = len(pixel_coords)
        factor = target_len // len(int_rgbas)
        int_rgbas = np.array([int_rgbas] * factor).reshape((target_len, rgba_len))

        on_screen_indices = self._on_screen_pixels(pixel_coords)
        pixel_coords = pixel_coords[on_screen_indices]
        int_rgbas = int_rgbas[on_screen_indices]

        height = self.target.settings.pixel_height
        width = self.target.settings.pixel_width
        flattener = np.array([1, width], dtype="int").reshape((2, 1))
        indices = np.dot(pixel_coords, flattener)[:, 0].astype("int")
        flattened = pixel_array.reshape((height * width, rgba_len))
        flattened[indices] = int_rgbas
        pixel_array[:, :] = flattened.reshape((height, width, rgba_len))

    def _display_image_mobjects(
        self,
        image_mobjects: list[AbstractImageMobject | ImageMobjectFromCamera],
    ) -> None:
        for image_mobject in image_mobjects:
            self._display_image_mobject(image_mobject)

    def _display_image_mobject(
        self,
        image_mobject: AbstractImageMobject | ImageMobjectFromCamera,
    ) -> None:
        source_pixels = (
            self.image_resolver(image_mobject)
            if isinstance(image_mobject, ImageMobjectFromCamera)
            else image_mobject.get_pixel_array()
        )
        if source_pixels is None:
            return
        sub_image = Image.fromarray(source_pixels, mode="RGBA")
        original_coords = np.array(
            [
                [0, 0],
                [sub_image.width, 0],
                [0, sub_image.height],
                [sub_image.width, sub_image.height],
            ],
        )
        target_coords = self._points_to_subpixel_coords(
            image_mobject,
            image_mobject.points,
        )
        int_target_coords = target_coords.astype(np.int64)
        shift_vector = np.array(
            [
                min(x for x, _ in int_target_coords),
                min(y for _, y in int_target_coords),
            ],
        )
        target_coords -= shift_vector
        int_target_coords -= shift_vector
        target_size = (
            max(x for x, _ in int_target_coords),
            max(y for _, y in int_target_coords),
        )
        if min(target_size) <= 0:
            return

        ordered_vertices = [target_coords[index] for index in (0, 1, 3, 2)]
        sides = [
            ordered_vertices[(index + 1) % 4] - ordered_vertices[index]
            for index in range(4)
        ]
        side_lengths = np.linalg.norm(sides, axis=1)
        longest_index = int(np.argmax(side_lengths))
        longest_side = sides[longest_index]
        longest_length = side_lengths[longest_index]
        if longest_length == 0:
            return
        previous_side = sides[(longest_index - 1) % 4]
        next_side = sides[(longest_index - 1) % 4]
        height_1 = abs(cross2d(longest_side, previous_side)) / longest_length
        height_2 = abs(cross2d(longest_side, next_side)) / longest_length
        if max(height_1, height_2) < 0.5:
            return

        homography_matrix = []
        for (x, y), (target_x, target_y) in zip(
            target_coords,
            original_coords,
            strict=True,
        ):
            homography_matrix.append(
                [x, y, 1, 0, 0, 0, -target_x * x, -target_x * y],
            )
            homography_matrix.append(
                [0, 0, 0, x, y, 1, -target_y * x, -target_y * y],
            )
        matrix = np.array(homography_matrix, dtype=np.float64)
        target = original_coords.reshape(8).astype(np.float64)
        try:
            coefficients = np.linalg.solve(matrix, target)
        except np.linalg.LinAlgError:
            return

        sub_image = sub_image.transform(
            size=target_size,
            method=Image.Transform.PERSPECTIVE,
            data=coefficients,
            resample=image_mobject.resampling_algorithm,
        )
        settings = self.target.settings
        full_image = Image.new(
            "RGBA",
            (settings.pixel_width, settings.pixel_height),
            (0, 0, 0, 0),
        )
        full_image.paste(
            sub_image,
            box=(
                int(shift_vector[0]),
                int(shift_vector[1]),
                int(shift_vector[0] + target_size[0]),
                int(shift_vector[1] + target_size[1]),
            ),
        )
        self._overlay_pil_image(self.target.pixels, full_image)

    def _overlay_rgba_array(
        self,
        pixel_array: RGBAPixelArray,
        new_array: RGBAPixelArray,
    ) -> None:
        self._overlay_pil_image(pixel_array, Image.fromarray(new_array))

    @staticmethod
    def _overlay_pil_image(
        pixel_array: RGBAPixelArray,
        image: Image.Image,
    ) -> None:
        pixel_array[:, :] = np.asarray(
            Image.alpha_composite(Image.fromarray(pixel_array), image),
            dtype=np.uint8,
        )

    def _points_to_subpixel_coords(
        self,
        mobject: Mobject,
        points: Point3D_Array,
    ) -> np.ndarray:
        points = self.camera.transform_points_pre_display(mobject, points)
        shifted_points = points - self.camera.frame_center
        settings = self.target.settings
        result = np.zeros((len(points), 2))
        result[:, 0] = (
            shifted_points[:, 0] * settings.pixel_width / self.camera.frame_width
            + settings.pixel_width / 2
        )
        result[:, 1] = (
            -shifted_points[:, 1] * settings.pixel_height / self.camera.frame_height
            + settings.pixel_height / 2
        )
        return result

    def _points_to_pixel_coords(
        self,
        mobject: Mobject,
        points: Point3D_Array,
    ) -> np.ndarray:
        return self._points_to_subpixel_coords(mobject, points).astype(np.int64)

    def _on_screen_pixels(self, pixel_coords: np.ndarray) -> np.ndarray:
        settings = self.target.settings
        return reduce(
            op.and_,
            [
                pixel_coords[:, 0] >= 0,
                pixel_coords[:, 0] < settings.pixel_width,
                pixel_coords[:, 1] >= 0,
                pixel_coords[:, 1] < settings.pixel_height,
            ],
        )

    def _adjusted_thickness(self, thickness: float) -> float:
        settings = self.target.settings
        base_sum = settings.base_pixel_height + settings.base_pixel_width
        target_sum = settings.pixel_height + settings.pixel_width
        return 1 + (thickness - 1) * base_sum / target_sum

    @staticmethod
    def _thickened_coordinates(
        pixel_coords: np.ndarray,
        thickness: float,
    ) -> np.ndarray:
        thickness = int(thickness)
        coordinate_range = list(
            range(-thickness // 2 + 1, thickness // 2 + 1),
        )
        nudges = np.array(list(it.product(coordinate_range, coordinate_range)))
        thickened = np.array([pixel_coords + nudge for nudge in nudges])
        return thickened.reshape((thickened.size // 2, 2))
