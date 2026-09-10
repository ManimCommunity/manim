"""Semantic camera state for the OpenGL rendering backend."""

from __future__ import annotations

import typing
from functools import cached_property
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from typing_extensions import override

from manim import config
from manim.constants import *
from manim.mobject.opengl.opengl_mobject import OpenGLMobject, OpenGLPoint
from manim.typing import MatrixMN, Point3D
from manim.utils import opengl
from manim.utils.paths import straight_path
from manim.utils.simple_functions import clip
from manim.utils.space_ops import (
    angle_of_vector,
    quaternion_from_angle_axis,
    quaternion_mult,
    rotation_matrix_transpose,
    rotation_matrix_transpose_from_quaternion,
)

if TYPE_CHECKING:
    from manim.typing import PathFuncType, Point3DLike, Vector3DLike
    from manim.utils.opengl import FlattenedMatrix4x4

__all__ = ["OpenGLCamera"]


class OpenGLCamera(OpenGLMobject):
    """
    An OpenGL-based camera for 3D scene rendering.


    Attributes
    ----------
    frame_shape : tuple[float, float]
        The width and height of the camera frame.
    center_point : np.ndarray
        The center point of the camera in 3D space.
    euler_angles : np.ndarray
        The Euler angles (theta, phi, gamma) representing the camera's orientation.
    focal_distance : float
        The focal distance of the camera.
    light_source_position : np.ndarray
        The position of the light source in 3D space.
    orthographic : bool
        Whether the camera uses orthographic projection instead of perspective.
    minimum_polar_angle : float
        The minimum polar angle for camera rotation.
    maximum_polar_angle : float
        The maximum polar angle for camera rotation.
    inverse_rotation_matrix : np.ndarray
        The inverse rotation matrix of the camera.
    """

    def __init__(
        self,
        frame_shape: tuple[float, float] | None = None,
        center_point: Point3DLike | None = None,
        # Theta, phi, gamma
        euler_angles: Point3DLike | None = None,
        focal_distance: float = 2.0,
        light_source_position: Point3DLike | None = None,
        orthographic: bool = False,
        minimum_polar_angle: float = -PI / 2,
        maximum_polar_angle: float = PI / 2,
        model_matrix: MatrixMN | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes an OpenGLCamera instance.

        Parameters
        ----------
        frame_shape : tuple[float, float], optional
            The width and height of the camera frame. If not provided, defaults to
            the global manim config values `frame_width` and `frame_height`.
        center_point : Point3DLike, optional
            The center point of the camera in 3D space.
            If not provided, defaults to the origin (0, 0, 0).
        euler_angles : Point3DLike, optional
            The Euler angles (theta, phi, gamma) representing the camera's orientation.
            If not provided, defaults to (0, 0, 0) (i.e., no rotation).
        focal_distance : float, optional
            The focal distance of the camera. Default is 2.0.
        light_source_position : Point3DLike, optional
            The position of the light source in 3D space.
            If not provided, defaults to (-10, 10, 10).
        orthographic : bool, optional
            Whether the camera uses orthographic projection instead of perspective.
            Default is False (perspective).
        minimum_polar_angle : float, optional
            The minimum polar angle in radian for camera rotation. Default is -π/2,
            i.e. no restriction.
        maximum_polar_angle : float, optional
            The maximum polar angle in radian for camera rotation. Default is π/2,
            i.e. no restriction.
        model_matrix : MatrixMN, optional
            The initial model matrix for the camera. If not provided, defaults to a
            translation matrix that positions the camera at (0, 0, 11).
        **kwargs : Any
            Additional keyword arguments passed to the OpenGLMobject constructor.
        """
        self.use_z_index = True
        self.orthographic = orthographic
        self.minimum_polar_angle = minimum_polar_angle
        self.maximum_polar_angle = maximum_polar_angle
        if self.orthographic:
            self.projection_matrix = opengl.orthographic_projection_matrix()
            self.unformatted_projection_matrix = opengl.orthographic_projection_matrix(
                format_=False,
            )
        else:
            self.projection_matrix = opengl.perspective_projection_matrix()
            self.unformatted_projection_matrix = opengl.perspective_projection_matrix(
                format_=False,
            )

        if frame_shape is None:
            self.frame_shape = (config["frame_width"], config["frame_height"])
        else:
            self.frame_shape = frame_shape

        if center_point is None:
            self.center_point = ORIGIN
        else:
            self.center_point = np.asarray(center_point, dtype=float)

        if model_matrix is None:
            model_matrix = opengl.translation_matrix(0, 0, 11)

        self.focal_distance = focal_distance

        self.light_source_position = np.asarray(
            light_source_position or [-10, 10, 10], dtype=float
        )

        self.light_source = OpenGLPoint(self.light_source_position)

        self.default_model_matrix = model_matrix
        super().__init__(model_matrix=model_matrix, should_render=False, **kwargs)

        euler_angles = np.asarray(euler_angles or [0, 0, 0], dtype=float)

        self.euler_angles: Point3D = euler_angles
        self.refresh_rotation_matrix()

    def get_position(self) -> Point3D:
        """Retrieve the camera's position in 3D space."""
        return self.model_matrix[:, 3][:3]

    def set_position(self, position: Point3D) -> Self:
        """Set the camera's position in 3D space."""
        self.model_matrix[:, 3][:3] = position
        return self

    @cached_property
    def formatted_view_matrix(self) -> FlattenedMatrix4x4:
        """The formatted view matrix for shader input."""
        return opengl.matrix_to_shader_input(self.unformatted_view_matrix)

    @cached_property
    def unformatted_view_matrix(self) -> MatrixMN:
        return typing.cast(MatrixMN, np.linalg.inv(self.model_matrix))

    def init_points(self) -> Self:
        """Initialize the camera's points based on frame shape and center point."""
        self.set_points([ORIGIN, LEFT, RIGHT, DOWN, UP])
        self.set_width(self.frame_shape[0], stretch=True)
        self.set_height(self.frame_shape[1], stretch=True)
        self.move_to(self.center_point)
        return self

    def to_default_state(self) -> Self:
        """Reset the camera to its default state
        (config frame size, centered at origin, no rotation).
        """
        self.center()
        self.set_height(config["frame_height"])
        self.set_width(config["frame_width"])
        self.set_euler_angles(0, 0, 0)
        self.model_matrix = self.default_model_matrix
        return self

    def refresh_rotation_matrix(self) -> Self:
        """Refresh the camera's inverse rotation matrix based on its Euler angles."""
        # Rotate based on camera orientation
        theta, phi, gamma = self.euler_angles
        quat = quaternion_mult(
            quaternion_from_angle_axis(theta, OUT, axis_normalized=True),
            quaternion_from_angle_axis(phi, RIGHT, axis_normalized=True),
            quaternion_from_angle_axis(gamma, OUT, axis_normalized=True),
        )
        self.inverse_rotation_matrix = rotation_matrix_transpose_from_quaternion(
            np.asarray(quat, dtype=float)
        )
        return self

    @override
    def rotate(
        self,
        angle: float,
        axis: Vector3DLike = OUT,
        about_point: Point3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Rotate the camera by a given angle around a specified axis.

        Parameters
        ----------
        angle : float
            The angle in radians to rotate the camera.
        axis : Vector3DLike, optional
            The axis around which to rotate the camera. Default is OUT (z-axis).
        about_point : Point3DLike, optional
            Ignored. For OpenGLCamera, rotation is always about the camera's center.

        **kwargs : Any
            Not used for OpenGLCamera. Passing additional keyword arguments
            has no effect.

        Returns
        -------
        Self
            The rotated camera instance. Returned for chaining.
        """
        curr_rot_T = self.inverse_rotation_matrix
        added_rot_T = rotation_matrix_transpose(angle, axis)
        new_rot_T = np.dot(curr_rot_T, added_rot_T)
        Fz = new_rot_T[2]
        phi = np.arccos(Fz[2])
        theta = angle_of_vector(Fz[:2]) + PI / 2
        partial_rot_T = np.dot(
            rotation_matrix_transpose(phi, RIGHT),
            rotation_matrix_transpose(theta, OUT),
        )
        gamma = angle_of_vector(np.dot(partial_rot_T, new_rot_T.T)[:, 0])
        self.set_euler_angles(theta, phi, gamma)
        return self

    def set_euler_angles(
        self,
        theta: float | None = None,
        phi: float | None = None,
        gamma: float | None = None,
    ) -> Self:
        """
        Set the camera's Euler angles [2]_ (theta, phi, gamma).

        Parameters
        ----------
        theta : float | None, optional
            The angle in radians for rotation around the OUT (z) axis.
            If None, the current theta value is retained.
        phi : float | None, optional
            The angle in radians for rotation around the RIGHT (x) axis.
            If None, the current phi value is retained.
        gamma : float | None, optional
            The angle in radians for rotation around the OUT (z) axis.
            If None, the current gamma value is retained.

        Returns
        -------
        Self
            The camera instance with updated Euler angles. Returned for chaining.

        See Also
        --------
        set_theta : Set the theta Euler angle.
        set_phi : Set the phi Euler angle.
        set_gamma : Set the gamma Euler angle.

        References
        ----------
        .. [2] Wikipedia, "Euler angles",
               https://en.wikipedia.org/wiki/Euler_angles
        """
        if theta is not None:
            self.euler_angles[0] = theta
        if phi is not None:
            self.euler_angles[1] = phi
        if gamma is not None:
            self.euler_angles[2] = gamma
        self.refresh_rotation_matrix()
        return self

    def set_theta(self, theta: float) -> Self:
        """
        Set the camera's theta Euler angle (in radians).

        See Also
        --------
        set_euler_angles : Set all Euler angles at once.
        set_phi : Set the phi Euler angle.
        set_gamma : Set the gamma Euler angle.
        """
        return self.set_euler_angles(theta=theta)

    def set_phi(self, phi: float) -> Self:
        """
        Set the camera's phi Euler angle (in radians).

        See Also
        --------
        set_euler_angles : Set all Euler angles at once.
        set_theta : Set the theta Euler angle.
        set_gamma : Set the gamma Euler angle.
        """
        return self.set_euler_angles(phi=phi)

    def set_gamma(self, gamma: float) -> Self:
        """
        Set the camera's gamma Euler angle (in radians).

        See Also
        --------
        set_euler_angles : Set all Euler angles at once.
        set_theta : Set the theta Euler angle.
        set_phi : Set the phi Euler angle.
        """
        return self.set_euler_angles(gamma=gamma)

    def increment_theta(self, dtheta: float) -> Self:
        """
        Increment the camera's theta Euler angle by a given amount (in radians).

        See Also
        --------
        set_euler_angles : Set all Euler angles at once.
        set_theta : Set the theta Euler angle.
        """
        self.euler_angles[0] += dtheta
        self.refresh_rotation_matrix()
        return self

    def increment_phi(self, dphi: float) -> Self:
        """
        Increment the camera's phi Euler angle by a given amount (in radians).

        See Also
        --------
        set_euler_angles : Set all Euler angles at once.
        set_phi : Set the phi Euler angle.
        """
        phi = self.euler_angles[1]
        new_phi = clip(phi + dphi, -PI / 2, PI / 2)
        self.euler_angles[1] = new_phi
        self.refresh_rotation_matrix()
        return self

    def increment_gamma(self, dgamma: float) -> Self:
        """
        Increment the camera's gamma Euler angle by a given amount (in radians).

        See Also
        --------
        set_euler_angles : Set all Euler angles at once.
        set_gamma : Set the gamma Euler angle.
        """
        self.euler_angles[2] += dgamma
        self.refresh_rotation_matrix()
        return self

    def get_shape(self) -> tuple[float, float]:
        """Retrieve the width and height of the camera frame."""
        return (self.get_width(), self.get_height())

    def get_center(self) -> Point3D:
        """
        Retrieve the center point of the camera in 3D space.

        Notes
        -----
        The center point is assumed to be the first point in the camera's points array.
        """
        # Assumes first point is at the center
        return typing.cast(Point3D, self.points[0])

    def get_width(self) -> float:
        """Retrieve the width of the camera frame."""
        points = self.points
        out = points[2, 0] - points[1, 0]
        return float(out)

    def get_height(self) -> float:
        """Retrieve the height of the camera frame."""
        points = self.points
        out = points[4, 1] - points[3, 1]
        return float(out)
        # return points[4, 1] - points[3, 1]

    def get_focal_distance(self) -> float:
        """Retrieve the focal distance of the camera."""
        return self.focal_distance * self.get_height()

    @override
    def interpolate(
        self,
        mobject1: OpenGLMobject,
        mobject2: OpenGLMobject,
        alpha: float,
        path_func: PathFuncType = straight_path(),
    ) -> Self:
        """Interpolate camera mobject state and refresh its rotation matrix."""
        super().interpolate(mobject1, mobject2, alpha, path_func)
        self.refresh_rotation_matrix()
        return self
