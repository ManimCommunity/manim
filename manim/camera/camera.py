from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from manim._config import config
from manim.mobject.opengl.opengl_mobject import OpenGLMobject

from ..constants import *
from ..utils.space_ops import normalize


class Camera(OpenGLMobject):
    def __init__(
        self,
        frame_shape: tuple[float, float] = (config.frame_width, config.frame_height),
        center_point: np.ndarray = ORIGIN,
        focal_dist_to_height: float = 2.0,
        **kwargs,
    ):
        self.frame_shape = frame_shape
        self.center_point = center_point
        self.focal_dist_to_height = focal_dist_to_height
        self.orientation = Rotation.identity().as_quat()
        super().__init__(**kwargs)

    def init_points(self) -> None:
        self.set_points([ORIGIN, LEFT, RIGHT, DOWN, UP])
        self.set_width(self.frame_shape[0], stretch=True)
        self.set_height(self.frame_shape[1], stretch=True)
        self.move_to(self.center_point)

    def set_orientation(self, rotation: Rotation):
        self.orientation = rotation.as_quat()
        return self

    def get_orientation(self):
        return Rotation.from_quat(self.orientation)

    def to_default_state(self):
        self.center()
        self.set_height(config.frame_width)
        self.set_width(config.frame_height)
        self.set_orientation(Rotation.identity())
        return self

    def get_euler_angles(self):
        return self.get_orientation().as_euler("zxz")[::-1]

    def get_theta(self):
        return self.get_euler_angles()[0]

    def get_phi(self):
        return self.get_euler_angles()[1]

    def get_gamma(self):
        return self.get_euler_angles()[2]

    def get_inverse_camera_rotation_matrix(self):
        return self.get_orientation().as_matrix().T

    def rotate(self, angle: float, axis: np.ndarray = OUT, **kwargs):  # type: ignore
        rot = Rotation.from_rotvec(axis * normalize(axis))  # type: ignore
        self.set_orientation(rot * self.get_orientation())

    def set_euler_angles(
        self,
        theta: float | None = None,
        phi: float | None = None,
        gamma: float | None = None,
        units: float = RADIANS,
    ):
        eulers = self.get_euler_angles()  # theta, phi, gamma
        for i, var in enumerate([theta, phi, gamma]):
            if var is not None:
                eulers[i] = var * units
        self.set_orientation(Rotation.from_euler("zxz", eulers[::-1]))
        return self

    def reorient(
        self,
        theta_degrees: float | None = None,
        phi_degrees: float | None = None,
        gamma_degrees: float | None = None,
    ):
        """
        Shortcut for set_euler_angles, defaulting to taking
        in angles in degrees
        """
        self.set_euler_angles(theta_degrees, phi_degrees, gamma_degrees, units=DEGREES)
        return self

    def set_theta(self, theta: float):
        return self.set_euler_angles(theta=theta)

    def set_phi(self, phi: float):
        return self.set_euler_angles(phi=phi)

    def set_gamma(self, gamma: float):
        return self.set_euler_angles(gamma=gamma)

    def increment_theta(self, dtheta: float):
        self.rotate(dtheta, OUT)
        return self

    def increment_phi(self, dphi: float):
        self.rotate(dphi, self.get_inverse_camera_rotation_matrix()[0])
        return self

    def increment_gamma(self, dgamma: float):
        self.rotate(dgamma, self.get_inverse_camera_rotation_matrix()[2])
        return self

    def set_focal_distance(self, focal_distance: float):
        self.focal_dist_to_height = focal_distance / self.get_height()
        return self

    def set_field_of_view(self, field_of_view: float):
        self.focal_dist_to_height = 2 * math.tan(field_of_view / 2)
        return self

    def get_shape(self):
        return (self.get_width(), self.get_height())

    def get_center(self) -> np.ndarray:
        # Assumes first point is at the center
        return self.points[0]

    def get_width(self) -> float:
        points = self.points
        return points[2, 0] - points[1, 0]

    def get_height(self) -> float:
        points = self.points
        return points[4, 1] - points[3, 1]

    def get_focal_distance(self) -> float:
        return self.focal_dist_to_height * self.get_height()  # type: ignore

    def get_field_of_view(self) -> float:
        return 2 * math.atan(self.focal_dist_to_height / 2)

    def get_implied_camera_location(self) -> np.ndarray:
        to_camera = self.get_inverse_camera_rotation_matrix()[2]
        dist = self.get_focal_distance()
        return self.get_center() + dist * to_camera
