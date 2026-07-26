"""A camera that controls the FOV, orientation, and position of the scene."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import numpy.typing as npt

from manim._config import config, logger
from manim.constants import *
from manim.mobject.opengl.opengl_mobject import InvisibleMobject
from manim.mobject.opengl.opengl_mobject import OpenGLMobject as Mobject
from manim.utils.paths import straight_path
from manim.utils.space_ops import rotation_matrix

if TYPE_CHECKING:
    from typing import Self

    from manim.typing import ManimFloat, MatrixMN, Point3D, Vector3D


class CameraOrientationConfig(TypedDict, total=False):
    theta: float | None
    phi: float | None
    gamma: float | None
    zoom: float | None
    focal_distance: float | None
    frame_center: Mobject | Sequence[float] | None


class Camera(Mobject, InvisibleMobject):
    def __init__(
        self,
        frame_shape: tuple[float, float] = (config.frame_width, config.frame_height),
        center_point: Point3D = ORIGIN,  # TODO: use Point3DLike
        focal_distance: float = 16.0,
        **kwargs: Any,
    ):
        self.initial_frame_shape = frame_shape
        self.center_point = center_point
        self.focal_distance = focal_distance
        self.set_euler_angles(theta=-TAU / 4, phi=0.0, gamma=0.0)
        self.ambient_rotation_updaters_dict: dict[Updater | None] = {
            "theta": None,
            "gamma": None,
            "phi": None,
        }
        self.precession_updater: Updater | None = None
        super().__init__(**kwargs)

    def init_points(self) -> None:
        self.set_points([ORIGIN, LEFT, RIGHT, DOWN, UP])
        self.set_width(self.initial_frame_shape[0], stretch=True)
        self.set_height(self.initial_frame_shape[1], stretch=True)
        self.move_to(self.center_point)

    def interpolate(
        self,
        mobject1: Mobject,
        mobject2: Mobject,
        alpha: float,
        path_func: PathFuncType = straight_path(),
    ) -> Self:
        """Interpolate the orientation of two cameras."""
        cam1: Camera = mobject1
        cam2: Camera = mobject2

        orientation1 = cam1.get_orientation()
        orientation2 = cam2.get_orientation()
        new_orientation = {
            key: path_func(orientation1[key], orientation2[key], alpha)
            for key in orientation1
        }
        return self.set_orientation(**new_orientation)

    def get_orientation(self) -> CameraOrientationConfig:
        return {
            "theta": self.get_theta(),
            "phi": self.get_phi(),
            "gamma": self.get_gamma(),
            "zoom": self.get_zoom(),
            "focal_distance": self.focal_distance,
            "frame_center": self.get_center(),
        }

    def set_orientation(
        self,
        theta: float | None = None,
        phi: float | None = None,
        gamma: float | None = None,
        zoom: float | None = None,
        focal_distance: float | None = None,
        frame_center: Mobject | Point3D | None = None,  # TODO: use Point3DLike
    ) -> Self:
        """This method sets the orientation of the camera in the scene.

        Parameters
        ----------
        theta
            The azimuthal angle i.e the angle that spins the camera around the Z_AXIS.
        phi
            The polar angle i.e the angle between Z_AXIS and Camera through ORIGIN in radians.
        gamma
            The rotation of the camera about the vector from the ORIGIN to the Camera.
        zoom
            The zoom factor of the scene.
        focal_distance
            The focal_distance of the Camera.
        frame_center
            The new center of the camera frame in cartesian coordinates.

        Returns
        -------
        :class:`Camera`
            The camera after applying all changes.
        """
        self.set_euler_angles(theta=theta, phi=phi, gamma=gamma)
        if focal_distance is not None:
            self.focal_distance = focal_distance
        if zoom is not None:
            self.set_zoom(zoom)
        if frame_center is not None:
            self.move_to(frame_center)
        return self

    def get_euler_angles(self) -> npt.NDArray[ManimFloat]:
        return np.array([self._theta, self._phi, self._gamma])

    def set_euler_angles(
        self,
        theta: float | None = None,
        phi: float | None = None,
        gamma: float | None = None,
    ) -> Self:
        if theta is not None:
            self.set_theta(theta)
        if phi is not None:
            self.set_phi(phi)
        if gamma is not None:
            self.set_gamma(gamma)
        return self

    def get_theta(self) -> float:
        """Get the angle theta along which the camera is rotated about the Z
        axis.

        Returns
        -------
        float
            The theta angle.
        """
        return self._theta

    def set_theta(self, theta: float) -> Self:
        """Set the angle theta by which the camera is rotated about the Z
        axis.

        Parameters
        ----------
        theta
            The new theta angle.

        Returns
        -------
        :class:`Camera`
            The camera after setting its theta angle.
        """
        self._theta = theta
        self._rotation_matrix = None
        # If we don't add TAU/4 (90°) to theta, the camera will be positioned
        # over the negative Y axis instead of the positive X axis.
        cos = np.cos(theta + TAU / 4)
        sin = np.sin(theta + TAU / 4)
        self._theta_z_matrix = np.array(
            [
                [cos, -sin, 0],
                [sin, cos, 0],
                [0, 0, 1],
            ]
        )
        return self

    def increment_theta(self, dtheta: float) -> Self:
        """Incremeet the angle theta by which the camera is rotated about the Z
        axis, by a given ``dtheta``.

        Parameters
        ----------
        dtheta
            The increment in the angle theta.

        Returns
        -------
        :class:`Camera`
            The camera after incrementing its theta angle.
        """
        return self.set_theta(self._theta + dtheta)

    def get_phi(self) -> float:
        """Get the angle phi between the camera and the Z axis.

        Returns
        -------
        float
            The phi angle.
        """
        return self._phi

    def set_phi(self, phi: float) -> Self:
        """Set the angle phi between the camera and the Z axis.

        Parameters
        ----------
        phi
            The new phi angle.

        Returns
        -------
        :class:`Camera`
            The camera after setting its phi angle.
        """
        self._phi = phi
        self._rotation_matrix = None
        cos = np.cos(phi)
        sin = np.sin(phi)
        self._phi_x_matrix = np.array(
            [
                [1, 0, 0],
                [0, cos, -sin],
                [0, sin, cos],
            ]
        )
        return self

    def increment_phi(self, dphi: float) -> Self:
        """Increment the angle phi between the camera and the Z axis by a given
        ``dphi``.

        Parameters
        ----------
        dphi
            The increment in the angle phi.

        Returns
        -------
        :class:`Camera`
            The camera after incrementing its phi angle.
        """
        return self.set_phi(self._phi + dgamma)

    def get_gamma(self) -> float:
        """Get the angle gamma by which the camera is rotated while standing on
        its current position.

        Returns
        -------
        float
            The gamma angle.
        """
        return self._gamma

    def set_gamma(self, gamma: float) -> Self:
        """Set the angle gamma by which the camera is rotated while standing on
        its current position.

        Parameters
        ----------
        gamma
            The new gamma angle.

        Returns
        -------
        :class:`Camera`
            The camera after setting its gamma angle.
        """
        self._gamma = gamma
        self._rotation_matrix = None
        cos = np.cos(gamma)
        sin = np.sin(gamma)
        self._gamma_z_matrix = np.array(
            [
                [cos, -sin, 0],
                [sin, cos, 0],
                [0, 0, 1],
            ]
        )
        return self

    def increment_gamma(self, dgamma: float) -> Self:
        """Increment the angle gamma by which the camera is rotated while
        standing on its current position, by an angle ``dgamma``.

        Parameters
        ----------
        dgamma
            The increment in the angle gamma.

        Returns
        -------
        :class:`Camera`
            The camera after incrementing its gamma angle.
        """
        return self.set_gamma(self._gamma + dgamma)

    def get_rotation_matrix(self) -> MatrixMN:
        r"""Get the current rotation matrix.

        In order to get the current rotation using the Euler angles:

            1. Rotate :math:`\gamma` along the Z axis (XY plane).
            2. Rotate :math:`\varphi` along the X axis (YZ plane).
            3. Rotate :math:`\theta` along the Z axis again (XY plane).

        See :meth:`Camera.rotate()` for more information.

        Returns
        -------
        MatrixMN
            The current 3x3 rotation matrix.
        """
        if self._rotation_matrix is None:
            self._rotation_matrix = (
                self._theta_z_matrix @ self._phi_x_matrix @ self._gamma_z_matrix
            )
        return self._rotation_matrix

    def get_inverse_rotation_matrix(self) -> MatrixMN:
        return self.get_rotation_matrix().T

    def set_focal_distance(self, focal_distance: float) -> Self:
        self.focal_distance = focal_distance
        return self

    # TODO: rotate is still unreliable. The Euler angles are automatically
    # standardized to (-TAU/2, TAU/2), leading to potentially unwanted behavior
    # when animating. Plus, if the camera is on the Z axis, which occurs when
    # phi is a multiple of TAU/2, the current implementation can only determine
    # theta + gamma, but not exactly theta or gamma yet.
    def rotate(self, angle: float, axis: Vector3D = OUT, **kwargs: Any) -> Self:
        r"""Rotate the camera in a given ``angle`` along the given ``axis``.

        After rotating the camera, the Euler angles must be recalculated, Given
        the Euler angles :math:`\theta`, :math:`\varphi` and :math:`\gamma`,
        the current rotation matrix is obtained in this way:

            1.  Rotate :math:`\gamma` along the Z axis (XY plane). This
                corresponds to multiplying by the following matrix:

                .. math::

                    R_z(\gamma) = \begin{pmatrix}
                        \cos(\gamma) & -\sin(\gamma) & 0 \\
                        \sin(\gamma) & \cos(\gamma) & 0 \\
                        0 & 0 & 1
                    \end{pmatrix}

            2.  Rotate :math:`\varphi` along the X axis (YZ plane). This
                corresponds to multiplying by the following matrix:

                .. math::

                    R_x(\varphi) = \begin{pmatrix}
                        1 & 0 & 0 \\
                        0 & \cos(\varphi) & -\sin(\gamma) \\
                        0 & \sin(\varphi) & \cos(\varphi)
                    \end{pmatrix}

            3.  Rotate :math:`\theta` along the Z axis again (XY plane). This
                corresponds to multiplying by the following matrix:

                .. math::

                    R_z(\theta) = \begin{pmatrix}
                        \cos(\theta) & -\sin(\theta) & 0 \\
                        \sin(\theta) & \cos(\theta) & 0 \\
                        0 & 0 & 1
                    \end{pmatrix}

        Applying these matrices in order, the final rotation matrix is:

        .. math::

            R = R_z(\theta) R_x(\varphi) R_z(\gamma) = \begin{pmatrix}
                \cos(\theta)\cos(\gamma) - \sin(\theta)\cos(\varphi)\cos(\gamma) & -\sin(\theta)\cos(varphi)\cos(\gamma) - \cos(\theta)\sin(\gamma) & \sin(\theta)\sin(\varphi) \\
                \cos(\theta)\cos(\varphi)\sin(\gamma) + \sin(\theta)\cos(\gamma) & \cos(\theta)\cos(varphi)\cos(\gamma) - \sin(\theta)\sin(\gamma) & -\cos(\theta)\sin(\varphi) \\
                \sin(\varphi)\sin(\gamma) & \sin(\varphi)\cos(\gamma) & \cos(\varphi)
            \end{pmatrix}

        From this matrix, if :math:`\sin(\varphi) \neq 0`, then it is possible
        to retrieve the Euler angles in the following way:

        .. math::

            \frac{R_{1,3}}{-R_{2,3}} = \frac{\sin(\theta)\sin(\varphi)}{\cos(\theta)\sin(\varphi)} = \tan(\theta) \quad &\Longrightarrow  \quad \theta = \text{atan2}(-R_{2,3}, R_{1,3}) \\
            \frac{\sqrt{R_{1,3}^2 + R_{2,3}^2}}{R_{3,3}} = \frac{\sqrt{\sin^2(\theta)\sin^2(\varphi) + \cos^2(\theta)\sin^2(\varphi)}}{\cos(\varphi)} = \tan(\varphi) \quad &\Longrightarrow  \quad \theta = \text{atan2}(\sqrt{R_{1,3}^2 + R_{2,3}^2}, R_{3,3}) \\
            \frac{R_{3,1}}{R_{3,2}} = \frac{\sin(\varphi)\sin(\gamma)}{\sin(\varphi)\cos(\gamma)} = \tan(\gamma) \quad &\Longrightarrow  \quad \theta = \text{atan2}(R_{3,2}, R_{3,1}) \\

        However, if :math:`\sin(\varphi) = 0`, then:

        .. math::
            \frac{R_{2,2}}{R_{1,1}} = \frac{\sin(\theta \pm \gamma)}{\cos(\theta \pm \gamma)} = \tan(\theta \pm \gamma) \quad \Longrightarrow \quad \theta \pm \gamma = \text{atan2}(R_{1,1}, R_{2,2})

        and currently there's no implemented way to exactly find :math:`\theta`
        and :math:`\gamma`, so this function sets :math:`\gamma = 0`.

        .. warning::

            This method is still unreliable. The Euler angles are automatically
            standardized to (-TAU/2, TAU/2), leading to potentially unwanted behavior
            when using :attr:`Mobject.animate`. Plus, if the camera is on
            the Z axis, which occurs when phi is a multiple of TAU/2, the current
            implementation can only determine theta +- gamma, but not exactly
            theta or gamma yet.

        Parameters
        ----------
        angle
            Angle of rotation.
        axis
            Axis of rotation.
        **kwargs
            Additional parameters which are required by
            :meth:`Mobject.rotate`.

        Returns
        -------
        :class:`Camera`
            The camera after the rotation.
        """
        logger.warning(
            "Using this method automatically standardizes the Euler angles "
            "theta, phi and gamma, which might result in unexpected behavior "
            "when animating the camera. If phi is 0° or 180°, this method "
            "is not able to determine exactly theta and gamma, because their "
            "axes are aligned. Therefore, in that case, gamma will be set to "
            "0°."
        )

        new_rot = rotation_matrix(angle, axis) @ self.get_rotation_matrix()

        # Recalculate theta, phi and gamma.
        cos_phi = new_rot[2, 2]
        # If phi is 0 or TAU/2, there's a gimbal lock and it's not trivial to
        # determine theta and gamma, only theta +- gamma.
        if cos_phi in [-1, 1]:
            cos_theta_pm_gamma = new_rot[0, 0]
            sin_theta_pm_gamma = new_rot[1, 0]
            theta_pm_gamma = np.arctan2(cos_theta_pm_gamma, sin_theta_pm_gamma)

            # TODO: based on the axis, maybe there is a way to recover theta and gamma.
            theta = theta_pm_gamma
            phi = 0.0 if cos_phi == 1 else TAU / 2
            gamma = 0.0
        else:
            sin_theta_sin_phi = new_rot[0, 2]
            cos_theta_sin_phi = -new_rot[1, 2]
            theta = np.arctan2(sin_theta_sin_phi, cos_theta_sin_phi)

            sin_phi = np.sqrt(sin_theta_sin_phi**2 + cos_theta_sin_phi**2)
            phi = np.arctan2(cos_phi, sin_phi)

            sin_phi_sin_gamma = new_rot[2, 0]
            sin_phi_cos_gamma = new_rot[2, 1]
            gamma = np.arctan2(sin_phi_cos_gamma, sin_phi_sin_gamma)

        self.set_euler_angles(theta=theta, phi=phi, gamma=gamma)
        self._rotation_matrix = new_rot
        return self

    def get_zoom(self) -> float:
        return config.frame_height / self.height

    def set_zoom(self, zoom: float) -> Self:
        scale_factor = config.frame_height / (zoom * self.height)
        return self.scale(scale_factor)

    def get_field_of_view(self) -> float:
        return 2 * math.atan(self.focal_distance / (2 * self.height))

    def set_field_of_view(self, field_of_view: float) -> Self:
        self.focal_distance = 2 * math.tan(field_of_view / 2) * self.height
        return self

    def get_frame_shape(self) -> tuple[float, float]:
        return (self.get_width(), self.get_height())

    def get_center(self, copy: bool = True) -> Point3D:
        # Assumes first point is at the center
        center = self.points[0]
        return center.copy() if copy else center

    def get_width(self) -> float:
        points = self.points
        return points[2, 0] - points[1, 0]

    def get_height(self) -> float:
        points = self.points
        return points[4, 1] - points[3, 1]

    def get_implied_camera_direction(self) -> Vector3D:
        """Use the rotation matrix given by the Euler angles theta, phi and
        gamma to calculate the direction along which the camera would be
        positioned if it had a physical position.

        Returns
        -------
        :class:`Vector3D`
            The direction along which the camera would be positioned if it had
            a physical position.
        """
        return self.get_rotation_matrix()[:, 2]

    def get_implied_camera_location(self) -> Point3D:
        """Use the Euler angles theta, phi and gamma, as well as the frame
        center and the focal distance, to calculate the point in which the
        camera would be positioned if it had a physical position.

        Returns
        -------
        :class:`Point3D`
            The point in which the camera would be positioned if it had a
            physical position.
        """
        to_camera = self.get_implied_camera_direction()
        return self.get_center() + self.focal_distance * to_camera

    # Movement methods

    def begin_ambient_rotation(self, rate: float = 0.02, about: str = "theta") -> Self:
        """Apply an updater to rotate the camera on every frame by modifying
        one of three Euler angles: "theta" (rotate about the Z axis), "phi"
        (modify the angle between the camera and the Z axis) or "gamma" (rotate
        the camera in its position while it's looking at the same point).

        Parameters
        ----------
        rate
            The rate at which the camera should rotate about the specified
            angle. A positive rate means counterclockwise rotation, and a
            negative rate means clockwise rotation.
        about
            One of 3 options: ["theta", "phi", "gamma"]. Defaults to "theta".

        Returns
        -------
        :class:`Camera`
            The camera after applying the rotation updater.
        """
        # TODO, use a ValueTracker for rate, so that it
        # can begin and end smoothly
        about: str = about.lower()
        self.stop_ambient_rotation(about=about)

        methods = {
            "theta": self.increment_theta,
            "phi": self.increment_phi,
            "gamma": self.increment_gamma,
        }
        if about not in methods:
            raise ValueError(f"Invalid ambient rotation angle '{about}'.")

        def ambient_rotation(mob: Camera, dt: float) -> Camera:
            methods[about](rate * dt)
            return mob

        self.add_updater(ambient_rotation)
        self.ambient_rotation_updaters_dict[about] = ambient_rotation
        return self

    def stop_ambient_rotation(self, about: str = "theta") -> Self:
        """Stop ambient camera rotation on the specified angle. If there's a
        corresponding ambient rotation updater applied on the camera, remove
        it.

        Parameters
        ----------
        about
            The Euler angle for which the rotation should stop. This angle can
            be "theta", "phi" or "gamma". Defaults to "theta".

        Returns
        -------
        :class:`Camera`
            The camera after applying the rotation updater.
        """
        about: str = about.lower()
        if about not in self.ambient_rotation_updaters_dict:
            raise ValueError(f"Invalid ambient rotation angle '{about}'.")

        updater = self.ambient_rotation_updaters_dict[about]
        if updater is not None:
            self.remove_updater(updater)
            self.ambient_rotation_updaters_dict[about] = None

        return self

    def begin_precession(
        self,
        rate: float = 1.0,
        radius: float = 0.2,
        origin_theta: float | None = None,
        origin_phi: float | None = None,
    ) -> Self:
        """Begin a camera precession by adding an updater. This precession
        consists of moving around the point given by ``origin_phi`` and
        ``origin_theta``, keeping the ``gamma`` Euler angle constant.

        Parameters
        ----------
        rate
            The rate at which the camera precession should operate.
        radius
            The precession radius.
        origin_phi
            The polar angle the camera should move around. If ``None``,
            defaults to the current ``phi`` angle.
        origin_theta
            The azimutal angle the camera should move around. If ``None``,
            defaults to the current ``theta`` angle.

        Returns
        -------
        :class:`Camera`
            The camera after applying the precession updater.
        """
        self.stop_precession()

        if origin_theta is None:
            origin_theta = self.get_theta()
        if origin_phi is None:
            origin_phi = self.get_phi()

        precession_angle = 0.0

        def precession(mob: Camera, dt: float) -> Camera:
            nonlocal precession_angle
            precession_angle += rate * dt
            dtheta = radius * np.sin(precession_angle)
            dphi = radius * np.cos(precession_angle)
            return mob.set_theta(origin_theta + dtheta).set_phi(origin_phi + dphi)

        self.add_updater(precession)
        self.precession_updater = precession
        return self

    def stop_precession(self) -> Self:
        """Remove the precession camera updater, if any.

        Returns
        -------
        :class:`Camera`
            The camera after removing the precession updater.
        """
        updater = self.precession_updater
        if updater is not None:
            self.remove_updater(updater)
            self.precession_updater = None
        return self
