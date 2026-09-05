"""Semantic camera implementations for the Cairo rendering backend."""

from __future__ import annotations

__all__ = ["Camera", "MovingCamera", "MultiCamera", "ThreeDCamera"]

import operator as op
from collections.abc import Callable, Iterable
from functools import reduce
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

from manim._config import config
from manim.constants import DEGREES, DOWN, LEFT, ORIGIN, OUT, RIGHT, UP
from manim.mobject.frame import ScreenRectangle
from manim.mobject.mobject import Mobject, _AnimationBuilder
from manim.mobject.three_d.three_d_utils import (
    get_3d_vmob_end_corner,
    get_3d_vmob_end_corner_unit_normal,
    get_3d_vmob_start_corner,
    get_3d_vmob_start_corner_unit_normal,
)
from manim.mobject.types.image_mobject import ImageMobjectFromCamera
from manim.mobject.types.point_cloud_mobject import Point
from manim.mobject.types.vectorized_mobject import VMobject
from manim.mobject.value_tracker import ValueTracker
from manim.utils.color import WHITE, ManimColor, ParsableManimColor, get_shaded_rgb
from manim.utils.family import extract_mobject_family_members
from manim.utils.iterables import list_difference_update
from manim.utils.space_ops import rotation_about_z, rotation_matrix

if TYPE_CHECKING:
    from manim.typing import (
        FloatRGBA_Array,
        MatrixMN,
        Point3D,
        Point3D_Array,
        Point3DLike,
    )


class Camera:
    """Describe the logical view used by a rendering backend.

    Camera owns an animatable frame, semantic background settings, display ordering,
    and pure point transformations. Raster targets, pixel dimensions, image buffers,
    and backend contexts belong to renderers. With no explicit frame dimensions,
    the configured logical width is preserved and height follows the configured
    output aspect ratio. Supplying one dimension derives the other from that ratio;
    supplying both dimensions or a custom frame preserves the requested geometry.
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
            if frame_width is None:
                resolved_width = (
                    float(config["frame_width"])
                    if frame_height is None
                    else frame_height * config.aspect_ratio
                )
            else:
                resolved_width = frame_width
            resolved_height = (
                resolved_width / config.aspect_ratio
                if frame_height is None
                else frame_height
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


class MovingCamera(Camera):
    """Named camera subclass with the standard movable-frame behavior."""


class MultiCamera(Camera):
    """Describe a primary view with camera-backed image mobjects."""

    def __init__(
        self,
        image_mobjects_from_cameras: Iterable[ImageMobjectFromCamera] | None = None,
        **kwargs: Any,
    ) -> None:
        self.image_mobjects_from_cameras: list[ImageMobjectFromCamera] = []
        if image_mobjects_from_cameras is not None:
            for image_mobject in image_mobjects_from_cameras:
                self.add_image_mobject_from_camera(image_mobject)
        super().__init__(**kwargs)

    def add_image_mobject_from_camera(
        self,
        image_mobject_from_camera: ImageMobjectFromCamera,
    ) -> None:
        """Register a camera-backed image for renderer-owned composition."""
        if not isinstance(image_mobject_from_camera.camera, Camera):
            raise TypeError("Nested Cairo views require a Cairo Camera.")
        self.image_mobjects_from_cameras.append(image_mobject_from_camera)

    def get_mobjects_indicating_movement(self) -> list[Mobject]:
        """Return controls whose movement changes a primary or nested view."""

        def collect(camera: Camera, visited: set[int]) -> list[Mobject]:
            if id(camera) in visited:
                return []
            visited.add(id(camera))
            if not isinstance(camera, MultiCamera):
                return camera.get_mobjects_indicating_movement()

            indicators = Camera.get_mobjects_indicating_movement(camera)
            for image_mobject in camera.image_mobjects_from_cameras:
                indicators.extend(collect(image_mobject.camera, visited))
            return indicators

        return collect(self, set())


class ThreeDCamera(Camera):
    def __init__(
        self,
        focal_distance: float = 20.0,
        shading_factor: float = 0.2,
        default_distance: float = 5.0,
        light_source_start_point: Point3DLike = 9 * DOWN + 7 * LEFT + 10 * OUT,
        should_apply_shading: bool = True,
        exponential_projection: bool = False,
        phi: float = 0,
        theta: float = -90 * DEGREES,
        gamma: float = 0,
        zoom: float = 1,
        **kwargs: Any,
    ):
        """Initializes the ThreeDCamera

        Parameters
        ----------
        *kwargs
            Any keyword argument of Camera.
        """
        super().__init__(**kwargs)
        self.focal_distance = focal_distance
        self.phi = phi
        self.theta = theta
        self.gamma = gamma
        self.zoom = zoom
        self.shading_factor = shading_factor
        self.default_distance = default_distance
        self.light_source_start_point = light_source_start_point
        self.light_source = Point(self.light_source_start_point)
        self.should_apply_shading = should_apply_shading
        self.exponential_projection = exponential_projection
        self.phi_tracker = ValueTracker(self.phi)
        self.theta_tracker = ValueTracker(self.theta)
        self.focal_distance_tracker = ValueTracker(self.focal_distance)
        self.gamma_tracker = ValueTracker(self.gamma)
        self.zoom_tracker = ValueTracker(self.zoom)
        self.fixed_orientation_mobjects: dict[Mobject, Callable[[], Point3D]] = {}
        self.fixed_in_frame_mobjects: set[Mobject] = set()
        self.reset_rotation_matrix()

    def _prepare_for_render(self) -> None:
        self.reset_rotation_matrix()

    def get_view_transform_center(self) -> Point3D:
        # project_points() already translates by frame_center.
        return ORIGIN.copy()

    def get_mobjects_indicating_movement(self) -> list[Mobject]:
        return [self.frame, self.light_source, *self.get_value_trackers()]

    def get_value_trackers(self) -> list[ValueTracker]:
        """A list of :class:`ValueTrackers <.ValueTracker>` of phi, theta, focal_distance,
        gamma and zoom.

        Returns
        -------
        list
            list of ValueTracker objects
        """
        return [
            self.phi_tracker,
            self.theta_tracker,
            self.focal_distance_tracker,
            self.gamma_tracker,
            self.zoom_tracker,
        ]

    def modified_rgbas(
        self, vmobject: VMobject, rgbas: FloatRGBA_Array
    ) -> FloatRGBA_Array:
        if not self.should_apply_shading:
            return rgbas
        if vmobject.shade_in_3d and (vmobject.get_num_points() > 0):
            light_source_point = self.light_source.points[0]
            if len(rgbas) < 2:
                shaded_rgbas = rgbas.repeat(2, axis=0)
            else:
                shaded_rgbas = np.array(rgbas[:2])
            shaded_rgbas[0, :3] = get_shaded_rgb(
                shaded_rgbas[0, :3],
                get_3d_vmob_start_corner(vmobject),
                get_3d_vmob_start_corner_unit_normal(vmobject),
                light_source_point,
            )
            shaded_rgbas[1, :3] = get_shaded_rgb(
                shaded_rgbas[1, :3],
                get_3d_vmob_end_corner(vmobject),
                get_3d_vmob_end_corner_unit_normal(vmobject),
                light_source_point,
            )
            return shaded_rgbas
        return rgbas

    def get_stroke_rgbas(
        self,
        vmobject: VMobject,
        background: bool = False,
    ) -> FloatRGBA_Array:  # NOTE : DocStrings From parent
        return self.modified_rgbas(vmobject, vmobject.get_stroke_rgbas(background))

    def get_fill_rgbas(
        self, vmobject: VMobject
    ) -> FloatRGBA_Array:  # NOTE : DocStrings From parent
        return self.modified_rgbas(vmobject, vmobject.get_fill_rgbas())

    def get_mobjects_to_display(
        self, *args: Any, **kwargs: Any
    ) -> list[Mobject]:  # NOTE : DocStrings From parent
        mobjects = super().get_mobjects_to_display(*args, **kwargs)
        rot_matrix = self.get_rotation_matrix()

        def z_key(mob: Mobject) -> float:
            if not (hasattr(mob, "shade_in_3d") and mob.shade_in_3d):
                return np.inf  # type: ignore[no-any-return]
            # Assign a number to a three dimensional mobjects
            # based on how close it is to the camera
            distance: float = np.dot(mob.get_z_index_reference_point(), rot_matrix.T)[2]
            return distance

        return sorted(mobjects, key=z_key)

    def get_phi(self) -> float:
        """Returns the Polar angle (the angle off Z_AXIS) phi.

        Returns
        -------
        float
            The Polar angle in radians.
        """
        return self.phi_tracker.get_value()

    def get_theta(self) -> float:
        """Returns the Azimuthal i.e the angle that spins the camera around the Z_AXIS.

        Returns
        -------
        float
            The Azimuthal angle in radians.
        """
        return self.theta_tracker.get_value()

    def get_focal_distance(self) -> float:
        """Returns focal_distance of the Camera.

        Returns
        -------
        float
            The focal_distance of the Camera in MUnits.
        """
        return self.focal_distance_tracker.get_value()

    def get_gamma(self) -> float:
        """Returns the rotation of the camera about the vector from the ORIGIN to the Camera.

        Returns
        -------
        float
            The angle of rotation of the camera about the vector
            from the ORIGIN to the Camera in radians
        """
        return self.gamma_tracker.get_value()

    def get_zoom(self) -> float:
        """Returns the zoom amount of the camera.

        Returns
        -------
        float
            The zoom amount of the camera.
        """
        return self.zoom_tracker.get_value()

    def set_phi(self, value: float) -> None:
        """Sets the polar angle i.e the angle between Z_AXIS and Camera through ORIGIN in radians.

        Parameters
        ----------
        value
            The new value of the polar angle in radians.
        """
        self.phi_tracker.set_value(value)

    def set_theta(self, value: float) -> None:
        """Sets the azimuthal angle i.e the angle that spins the camera around Z_AXIS in radians.

        Parameters
        ----------
        value
            The new value of the azimuthal angle in radians.
        """
        self.theta_tracker.set_value(value)

    def set_focal_distance(self, value: float) -> None:
        """Sets the focal_distance of the Camera.

        Parameters
        ----------
        value
            The focal_distance of the Camera.
        """
        self.focal_distance_tracker.set_value(value)

    def set_gamma(self, value: float) -> None:
        """Sets the angle of rotation of the camera about the vector from the ORIGIN to the Camera.

        Parameters
        ----------
        value
            The new angle of rotation of the camera.
        """
        self.gamma_tracker.set_value(value)

    def set_zoom(self, value: float) -> None:
        """Sets the zoom amount of the camera.

        Parameters
        ----------
        value
            The zoom amount of the camera.
        """
        self.zoom_tracker.set_value(value)

    def reset_rotation_matrix(self) -> None:
        """Sets the value of self.rotation_matrix to
        the matrix corresponding to the current position of the camera
        """
        self.rotation_matrix = self.generate_rotation_matrix()

    def get_rotation_matrix(self) -> MatrixMN:
        """Returns the matrix corresponding to the current position of the camera.

        Returns
        -------
        np.array
            The matrix corresponding to the current position of the camera.
        """
        return self.rotation_matrix

    def generate_rotation_matrix(self) -> MatrixMN:
        """Generates a rotation matrix based off the current position of the camera.

        Returns
        -------
        np.array
            The matrix corresponding to the current position of the camera.
        """
        phi = self.get_phi()
        theta = self.get_theta()
        gamma = self.get_gamma()
        matrices = [
            rotation_about_z(-theta - 90 * DEGREES),
            rotation_matrix(-phi, RIGHT),
            rotation_about_z(gamma),
        ]
        result = np.identity(3)
        for matrix in matrices:
            result = np.dot(matrix, result)
        return result

    def project_points(self, points: Point3D_Array) -> Point3D_Array:
        """Applies the current rotation_matrix as a projection
        matrix to the passed array of points.

        Parameters
        ----------
        points
            The list of points to project.

        Returns
        -------
        np.array
            The points after projecting.
        """
        frame_center = self.frame_center
        focal_distance = self.get_focal_distance()
        zoom = self.get_zoom()
        rot_matrix = self.get_rotation_matrix()

        points = points - frame_center
        points = np.dot(points, rot_matrix.T)
        zs = points[:, 2]
        for i in 0, 1:
            if self.exponential_projection:
                # Proper projection would involve multiplying
                # x and y by d / (d-z).  But for points with high
                # z value that causes weird artifacts, and applying
                # the exponential helps smooth it out.
                factor = np.exp(zs / focal_distance)
                lt0 = zs < 0
                factor[lt0] = focal_distance / (focal_distance - zs[lt0])
            else:
                factor = focal_distance / (focal_distance - zs)
                factor[(focal_distance - zs) < 0] = 10**6
            points[:, i] *= factor * zoom
        return points

    def project_point(self, point: Point3D) -> Point3D:
        """Applies the current rotation_matrix as a projection
        matrix to the passed point.

        Parameters
        ----------
        point
            The point to project.

        Returns
        -------
        np.array
            The point after projection.
        """
        return self.project_points(point.reshape((1, 3)))[0, :]

    def transform_points_pre_display(
        self,
        mobject: Mobject,
        points: Point3D_Array,
    ) -> Point3D_Array:  # TODO: Write Docstrings for this Method.
        points = super().transform_points_pre_display(mobject, points)
        fixed_orientation = mobject in self.fixed_orientation_mobjects
        fixed_in_frame = mobject in self.fixed_in_frame_mobjects

        if fixed_in_frame:
            return points
        if fixed_orientation:
            center_func = self.fixed_orientation_mobjects[mobject]
            center = center_func()
            new_center = self.project_point(center)
            return points + (new_center - center)
        else:
            return self.project_points(points)

    def add_fixed_orientation_mobjects(
        self,
        *mobjects: Mobject,
        use_static_center_func: bool = False,
        center_func: Callable[[], Point3D] | None = None,
    ) -> None:
        """This method allows the mobject to have a fixed orientation,
        even when the camera moves around.
        E.G If it was passed through this method, facing the camera, it
        will continue to face the camera even as the camera moves.
        Highly useful when adding labels to graphs and the like.

        Parameters
        ----------
        *mobjects
            The mobject whose orientation must be fixed.
        use_static_center_func
            Whether or not to use the function that takes the mobject's
            center as centerpoint, by default False
        center_func
            The function which returns the centerpoint
            with respect to which the mobject will be oriented, by default None
        """

        # This prevents the computation of mobject.get_center
        # every single time a projection happens
        def get_static_center_func(mobject: Mobject) -> Callable[[], Point3D]:
            point = mobject.get_center()
            return lambda: point

        for mobject in mobjects:
            if center_func:
                func = center_func
            elif use_static_center_func:
                func = get_static_center_func(mobject)
            else:
                func = mobject.get_center
            for submob in mobject.get_family():
                self.fixed_orientation_mobjects[submob] = func

    def add_fixed_in_frame_mobjects(self, *mobjects: Mobject) -> None:
        """This method allows the mobject to have a fixed position,
        even when the camera moves around.
        E.G If it was passed through this method, at the top of the frame, it
        will continue to be displayed at the top of the frame.

        Highly useful when displaying Titles or formulae or the like.

        Parameters
        ----------
        **mobjects
            The mobject to fix in frame.
        """
        for mobject in extract_mobject_family_members(mobjects):
            self.fixed_in_frame_mobjects.add(mobject)

    def remove_fixed_orientation_mobjects(self, *mobjects: Mobject) -> None:
        """If a mobject was fixed in its orientation by passing it through
        :meth:`.add_fixed_orientation_mobjects`, then this undoes that fixing.
        The Mobject will no longer have a fixed orientation.

        Parameters
        ----------
        mobjects
            The mobjects whose orientation need not be fixed any longer.
        """
        for mobject in extract_mobject_family_members(mobjects):
            if mobject in self.fixed_orientation_mobjects:
                del self.fixed_orientation_mobjects[mobject]

    def remove_fixed_in_frame_mobjects(self, *mobjects: Mobject) -> None:
        """If a mobject was fixed in frame by passing it through
        :meth:`.add_fixed_in_frame_mobjects`, then this undoes that fixing.
        The Mobject will no longer be fixed in frame.

        Parameters
        ----------
        mobjects
            The mobjects which need not be fixed in frame any longer.
        """
        for mobject in extract_mobject_family_members(mobjects):
            if mobject in self.fixed_in_frame_mobjects:
                self.fixed_in_frame_mobjects.remove(mobject)
