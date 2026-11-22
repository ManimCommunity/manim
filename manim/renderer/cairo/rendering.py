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
from manim.renderer._paint import paint_colors
from manim.renderer._view import (
    display_points,
    is_3d,
    shaded_rgbas,
    sort_for_display,
    view_transform_center,
)
from manim.renderer.extract import Extractor, extract
from manim.renderer.snapshot import (
    RCameraView,
    RImage,
    RMesh,
    RNode,
    RPaint,
    RPath,
    RPoints,
    RStroke,
)
from manim.utils._path_points import (
    points_equal_2d,
    subpath_split_indices,
)
from manim.utils.space_ops import cross2d

from ._mesh_faces import mesh_faces
from .target import _CairoRenderTarget, _PathCache

if TYPE_CHECKING:
    from manim.renderer.protocol import CameraState
    from manim.typing import (
        FloatRGBA_Array,
        Point3D_Array,
        RGBAPixelArray,
    )


_LINE_JOIN_MAP = {
    LineJointType.AUTO: cairo.LineJoin.MITER,  # cairo default; TODO: this could be improved
    LineJointType.ROUND: cairo.LineJoin.ROUND,
    LineJointType.BEVEL: cairo.LineJoin.BEVEL,
    LineJointType.MITER: cairo.LineJoin.MITER,
}

_CAP_STYLE_MAP = {
    CapStyleType.AUTO: cairo.LineCap.BUTT,  # cairo default; TODO: this could be improved
    CapStyleType.ROUND: cairo.LineCap.ROUND,
    CapStyleType.BUTT: cairo.LineCap.BUTT,
    CapStyleType.SQUARE: cairo.LineCap.SQUARE,
}

#: Returns the pixels of a camera view, by its node.
_ImageResolver = Callable[[RNode], np.ndarray | None]


class _CairoDrawingContext:
    """Draw mobjects into one renderer-owned target using one semantic camera."""

    def __init__(
        self,
        *,
        camera: CameraState,
        target: _CairoRenderTarget,
        image_resolver: _ImageResolver,
        use_z_index: bool = True,
        extractor: Extractor | None = None,
    ) -> None:
        self.camera = camera
        self.use_z_index = use_z_index
        self.extractor = extractor
        self.target = target
        self.image_resolver = image_resolver
        self._path_cache: _PathCache | None = None

    def nodes(
        self,
        mobjects: Iterable[Mobject],
        *,
        include_submobjects: bool = True,
        excluded_mobjects: list[Mobject] | None = None,
    ) -> list[RNode]:
        """The snapshots to draw for ``mobjects``, in drawing order: meshes as
        their faces, sorted by depth for a perspective view.
        """
        nodes = []
        take = extract if self.extractor is None else self.extractor.extract
        for node in take(
            mobjects,
            include_submobjects=include_submobjects,
            use_z_index=self.use_z_index,
            excluded_mobjects=excluded_mobjects or (),
        ):
            if isinstance(node.payload, RMesh):
                # A mesh is drawn as its faces, each a path.
                nodes.extend(mesh_faces(node, self.camera))
            else:
                nodes.append(node)
        return sort_for_display(self.camera, nodes)

    def draw(
        self,
        mobjects: Iterable[Mobject],
        *,
        include_submobjects: bool = True,
        excluded_mobjects: list[Mobject] | None = None,
    ) -> None:
        self.start_frame()
        self.draw_nodes(
            self.nodes(
                mobjects,
                include_submobjects=include_submobjects,
                excluded_mobjects=excluded_mobjects,
            )
        )

    def start_frame(self) -> None:
        """Begin a frame of the target: paths cached in the frame before last and
        not drawn since are dropped.
        """
        self.target._ensure_open()
        # Paths can be cached only where points are drawn where they are: a flat
        # view (a perspective view projects them anew in each frame).
        self._path_cache = None if is_3d(self.camera) else self.target.path_cache
        if self._path_cache is not None:
            self._path_cache.next_generation()

    def draw_nodes(self, nodes: list[RNode]) -> None:
        """Draw ``nodes`` in order over what the target holds (after
        :meth:`start_frame`).
        """
        self.target._ensure_open()
        for payload_type, group in it.groupby(nodes, lambda node: type(node.payload)):
            if payload_type is RPath:
                self._display_paths(list(group))
            elif payload_type is RPoints:
                self._display_point_clouds(group)
            else:
                self._display_images(group)

    def _display_paths(self, nodes: list[RNode]) -> None:
        for mask, batch in it.groupby(nodes, _mask):
            if mask:
                self._display_masked_paths(mask, list(batch))
            else:
                self._display_unmasked_paths(batch)

    def _display_unmasked_paths(self, nodes: Iterable[RNode]) -> None:
        context = self.target.get_context(self.camera)
        for node in nodes:
            self._display_path(node, context)

    def _display_path(self, node: RNode, context: cairo.Context) -> None:
        path: RPath = node.payload  # type: ignore[assignment]
        self._set_cairo_context_path(context, node, path)
        if path.background_stroke is not None:
            self._apply_stroke(context, node, path, path.background_stroke)
        self._apply_fill(context, node, path)
        self._apply_stroke(context, node, path, path.stroke)

    def _set_cairo_context_path(
        self, context: cairo.Context, node: RNode, path: RPath
    ) -> None:
        # Clear the previous path first: fills and strokes use the *_preserve
        # variants, so an early return must not leave another node's path
        # behind to be painted in this node's style.
        context.new_path()
        cache = self._path_cache
        key = node.key
        if cache is not None and key is not None:
            matrix = context.get_matrix()
            view = (matrix.xx, matrix.yy, matrix.x0, matrix.y0)
            cached = cache.get(key, view)
            if cached is not None:
                context.append_path(cached)
                return
            if self._build_cairo_context_path(context, node, path):
                cache.put(key, view, context.copy_path())
            return
        self._build_cairo_context_path(context, node, path)

    def _build_cairo_context_path(
        self, context: cairo.Context, node: RNode, path: RPath
    ) -> bool:
        """Set the context's path to the node's path; return whether it did."""
        points = display_points(self.camera, node, path.points)
        if len(points) == 0:
            return False

        nppcc = 4
        split_indices = subpath_split_indices(points, path.tolerance, n_dims=2)
        if len(split_indices) == 0:
            return False

        points_xy = points[:, :2].ravel()
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
            if points_equal_2d(
                points[start_index],
                points[end_index - 1],
                path.tolerance,
            ):
                close_path()
        return True

    def _set_cairo_context_color(
        self,
        context: cairo.Context,
        paint: RPaint,
        node: RNode,
        path: RPath,
    ) -> None:
        rgbas, ends = paint_colors(
            paint, path, lambda path, rgbas: shaded_rgbas(self.camera, path, rgbas)
        )
        if ends is None:
            context.set_source_rgba(*rgbas[0][2::-1], rgbas[0][3])
            return
        points = display_points(self.camera, node, ends)
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

    def _apply_fill(self, context: cairo.Context, node: RNode, path: RPath) -> None:
        self._set_cairo_context_color(context, path.fill, node, path)
        context.fill_preserve()

    def _apply_stroke(
        self,
        context: cairo.Context,
        node: RNode,
        path: RPath,
        stroke: RStroke,
    ) -> None:
        if stroke.width == 0:
            return
        self._set_cairo_context_color(context, stroke.paint, node, path)
        context.set_line_width(
            stroke.width * self.target.settings.cairo_line_width_multiple,
        )
        context.set_line_join(_LINE_JOIN_MAP[stroke.joint])
        context.set_line_cap(_CAP_STYLE_MAP[stroke.cap])
        context.stroke_preserve()

    def _display_masked_paths(
        self, image: Image.Image | str, nodes: list[RNode]
    ) -> None:
        """Draw paths that share one mask image: their colors times the image."""
        scratch = self.target.get_scratch_target()
        scratch.clear()
        scratch_context = _CairoDrawingContext(
            camera=self.camera,
            target=scratch,
            image_resolver=self.image_resolver,
            use_z_index=self.use_z_index,
        )
        scratch_context._display_unmasked_paths(nodes)
        background = self.target.get_background_image(image)
        colored = np.asarray(
            background * scratch.pixels.astype(float) / 255,
            dtype=np.uint8,
        )
        self._overlay_rgba_array(self.target.pixels, colored)

    def _display_point_clouds(self, nodes: Iterable[RNode]) -> None:
        for node in nodes:
            cloud: RPoints = node.payload  # type: ignore[assignment]
            self._display_point_cloud(
                node,
                cloud.points,
                cloud.rgbas,
                self._adjusted_thickness(cloud.stroke_width),
            )

    def _display_point_cloud(
        self,
        node: RNode,
        points: Point3D_Array,
        rgbas: FloatRGBA_Array,
        thickness: float,
    ) -> None:
        if len(points) == 0:
            return
        pixel_coords = self._points_to_pixel_coords(node, points)
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

    def _display_images(self, nodes: Iterable[RNode]) -> None:
        for node in nodes:
            self._display_image(node)

    def _display_image(self, node: RNode) -> None:
        image: RImage | RCameraView = node.payload  # type: ignore[assignment]
        source_pixels = (
            self.image_resolver(node)
            if isinstance(image, RCameraView)
            else image.pixel_array
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
        target_coords = self._points_to_subpixel_coords(node, image.points)
        int_target_coords = target_coords.astype(np.int64)
        settings = self.target.settings
        # Only rasterize the part of the image that lies inside the frame: the
        # cost of the perspective transform below grows with the size of its
        # output, which can exceed the frame by orders of magnitude when the
        # camera is zoomed in on the image.
        shift_vector = np.maximum(int_target_coords.min(axis=0), 0)
        upper_bounds = np.minimum(
            int_target_coords.max(axis=0),
            (settings.pixel_width, settings.pixel_height),
        )
        target_size = (
            int(upper_bounds[0] - shift_vector[0]),
            int(upper_bounds[1] - shift_vector[1]),
        )
        if min(target_size) <= 0:
            return
        target_coords -= shift_vector

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
        next_side = sides[(longest_index + 1) % 4]
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
            resample=image.resampling_algorithm,
        )
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
        node: RNode,
        points: Point3D_Array,
    ) -> np.ndarray:
        points = display_points(self.camera, node, points)
        # Use the same view center as the Cairo context transform, so that
        # pixel-placed images and point clouds line up with vectorized mobjects
        # (ThreeDCamera.project_points() has already removed frame_center).
        shifted_points = points - view_transform_center(self.camera)
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
        node: RNode,
        points: Point3D_Array,
    ) -> np.ndarray:
        return self._points_to_subpixel_coords(node, points).astype(np.int64)

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


def _mask(node: RNode) -> Any:
    path: RPath = node.payload  # type: ignore[assignment]
    return path.mask


def reusable_boundary(nodes: list[RNode], count: int) -> int:
    """The largest ``k <= count`` such that drawing ``nodes[:k]`` and then the rest
    gives the same pixels as drawing all at once.

    Paths with one mask image are drawn together, as a mask over the image; a run
    of them is not split.
    """
    image = _background_image_of
    while 0 < count < len(nodes):
        before, after = image(nodes[count - 1]), image(nodes[count])
        if before is None or before is not after:
            break
        count -= 1
    return count


def _background_image_of(node: RNode) -> Any:
    payload = node.payload
    return payload.mask if isinstance(payload, RPath) else None
