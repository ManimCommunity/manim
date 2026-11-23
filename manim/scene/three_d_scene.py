"""A scene suitable for rendering three-dimensional objects and animations."""

from __future__ import annotations

__all__ = ["ThreeDScene"]


from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from manim.mobject.value_tracker import ValueTracker

from ..animation.animation import Animation
from ..camera import ThreeDCamera
from ..constants import DEGREES
from ..light import DEFAULT_LIGHT_POSITION, PointLight
from ..mobject.mobject import Mobject
from ..scene.scene import Scene


class ThreeDScene(Scene):
    """
    This is a Scene, with special configurations and properties that
    make it suitable for Three Dimensional Scenes.
    """

    def __init__(
        self,
        camera_class=ThreeDCamera,
        ambient_camera_rotation=None,
        default_angled_camera_orientation_kwargs=None,
        **kwargs,
    ):
        self.ambient_camera_rotation = ambient_camera_rotation
        if default_angled_camera_orientation_kwargs is None:
            default_angled_camera_orientation_kwargs = {
                "phi": 70 * DEGREES,
                "theta": -135 * DEGREES,
            }
        self.default_angled_camera_orientation_kwargs = (
            default_angled_camera_orientation_kwargs
        )
        super().__init__(camera_class=camera_class, **kwargs)
        # The camera methods of this class control a 3D orientation.
        if not isinstance(self.camera, ThreeDCamera):
            raise TypeError(
                f"Expected a ThreeDCamera, but got {type(self.camera).__name__}"
            )
        #: The scene's light (see :attr:`.Scene.light`). Remove it
        #: (``self.light = None``) for no shading.
        self.light = PointLight(DEFAULT_LIGHT_POSITION)

    def set_camera_orientation(
        self,
        phi: float | None = None,
        theta: float | None = None,
        gamma: float | None = None,
        zoom: float | None = None,
        focal_distance: float | None = None,
        frame_center: Mobject | Sequence[float] | None = None,
        **kwargs,
    ):
        """
        This method sets the orientation of the camera in the scene.

        Parameters
        ----------
        phi
            The polar angle i.e the angle between Z_AXIS and Camera through ORIGIN in radians.

        theta
            The azimuthal angle i.e the angle that spins the camera around the Z_AXIS.

        focal_distance
            The focal_distance of the Camera.

        gamma
            The rotation of the camera about the vector from the ORIGIN to the Camera.

        zoom
            The zoom factor of the scene.

        frame_center
            The new center of the camera frame in cartesian coordinates.

        """
        if phi is not None:
            self.camera.set_phi(phi)
        if theta is not None:
            self.camera.set_theta(theta)
        if focal_distance is not None:
            self.camera.set_focal_distance(focal_distance)
        if gamma is not None:
            self.camera.set_gamma(gamma)
        if zoom is not None:
            self.camera.set_zoom(zoom)
        if frame_center is not None:
            self.camera.frame_center = frame_center

    def begin_ambient_camera_rotation(
        self, rate: float = 0.02, about: str = "theta"
    ) -> None:
        """
        This method begins an ambient rotation of the camera about the Z_AXIS,
        in the anticlockwise direction

        Parameters
        ----------
        rate
            The rate at which the camera should rotate about the Z_AXIS.
            Negative rate means clockwise rotation.
        about
            one of 3 options: ["theta", "phi", "gamma"]. defaults to theta.
        """
        # TODO, use a ValueTracker for rate, so that it
        # can begin and end smoothly
        about: str = about.lower()
        try:
            if isinstance(self.camera, ThreeDCamera):
                trackers = {
                    "theta": self.camera.theta_tracker,
                    "phi": self.camera.phi_tracker,
                    "gamma": self.camera.gamma_tracker,
                }
                x: ValueTracker = trackers[about]
                x.add_updater(lambda m, dt: x.increment_value(rate * dt))
                self.add(x)
        except Exception as e:
            raise ValueError("Invalid ambient rotation angle.") from e

    def stop_ambient_camera_rotation(self, about="theta"):
        """This method stops all ambient camera rotation."""
        about: str = about.lower()
        try:
            if isinstance(self.camera, ThreeDCamera):
                trackers = {
                    "theta": self.camera.theta_tracker,
                    "phi": self.camera.phi_tracker,
                    "gamma": self.camera.gamma_tracker,
                }
                x: ValueTracker = trackers[about]
                x.clear_updaters()
                self.remove(x)
        except Exception as e:
            raise ValueError("Invalid ambient rotation angle.") from e

    def begin_3dillusion_camera_rotation(
        self,
        rate: float = 1,
        origin_phi: float | None = None,
        origin_theta: float | None = None,
    ):
        """
        This method creates a 3D camera rotation illusion around
        the current camera orientation.

        Parameters
        ----------
        rate
            The rate at which the camera rotation illusion should operate.
        origin_phi
            The polar angle the camera should move around. Defaults
            to the current phi angle.
        origin_theta
            The azimutal angle the camera should move around. Defaults
            to the current theta angle.
        """
        if origin_theta is None:
            origin_theta = self.camera.theta_tracker.get_value()
        if origin_phi is None:
            origin_phi = self.camera.phi_tracker.get_value()

        val_tracker_theta = ValueTracker(0)

        def update_theta(m, dt):
            val_tracker_theta.increment_value(dt * rate)
            val_for_left_right = 0.2 * np.sin(val_tracker_theta.get_value())
            return m.set_value(origin_theta + val_for_left_right)

        self.camera.theta_tracker.add_updater(update_theta)
        self.add(self.camera.theta_tracker)

        val_tracker_phi = ValueTracker(0)

        def update_phi(m, dt):
            val_tracker_phi.increment_value(dt * rate)
            val_for_up_down = 0.1 * np.cos(val_tracker_phi.get_value()) - 0.1
            return m.set_value(origin_phi + val_for_up_down)

        self.camera.phi_tracker.add_updater(update_phi)
        self.add(self.camera.phi_tracker)

    def stop_3dillusion_camera_rotation(self):
        """This method stops all illusion camera rotations."""
        self.camera.theta_tracker.clear_updaters()
        self.remove(self.camera.theta_tracker)
        self.camera.phi_tracker.clear_updaters()
        self.remove(self.camera.phi_tracker)

    def move_camera(
        self,
        phi: float | None = None,
        theta: float | None = None,
        gamma: float | None = None,
        zoom: float | None = None,
        focal_distance: float | None = None,
        frame_center: Mobject | Sequence[float] | None = None,
        added_anims: Iterable[Animation] = [],
        **kwargs,
    ):
        """
        This method animates the movement of the camera
        to the given spherical coordinates.

        Parameters
        ----------
        phi
            The polar angle i.e the angle between Z_AXIS and Camera through ORIGIN in radians.

        theta
            The azimuthal angle i.e the angle that spins the camera around the Z_AXIS.

        focal_distance
            The radial focal_distance between ORIGIN and Camera.

        gamma
            The rotation of the camera about the vector from the ORIGIN to the Camera.

        zoom
            The zoom factor of the camera.

        frame_center
            The new center of the camera frame in cartesian coordinates.

        added_anims
            Any other animations to be played at the same time.

        """
        anims = []

        if isinstance(self.camera, ThreeDCamera):
            value_tracker_pairs = [
                (phi, self.camera.phi_tracker),
                (theta, self.camera.theta_tracker),
                (focal_distance, self.camera.focal_distance_tracker),
                (gamma, self.camera.gamma_tracker),
                (zoom, self.camera.zoom_tracker),
            ]
            for value, tracker in value_tracker_pairs:
                if value is not None:
                    anims.append(tracker.animate.set_value(value))
            if frame_center is not None:
                anims.append(self.camera.frame.animate.move_to(frame_center))

        frame_was_shown = (
            isinstance(self.camera, ThreeDCamera) and self.camera.frame in self.mobjects
        )
        self.play(*anims + added_anims, **kwargs)

        # Animating the camera frame adds it to the scene; it is a control, not
        # scene content, unless it was shown before.
        if (
            frame_center is not None
            and isinstance(self.camera, ThreeDCamera)
            and not frame_was_shown
        ):
            self.remove(self.camera.frame)

    def add_fixed_orientation_mobjects(self, *mobjects: Mobject, **kwargs: Any) -> None:
        """
        This method is used to prevent the rotation and tilting
        of mobjects as the camera moves around. The mobject can
        still move in the x,y,z directions, but will always be
        at the angle (relative to the camera) that it was at
        when it was passed through this method.)

        Parameters
        ----------
        *mobjects
            The Mobject(s) whose orientation must be fixed.

        **kwargs
            Some valid kwargs are
                use_static_center_func : bool
                center_func : function
        """
        if isinstance(self.camera, ThreeDCamera):
            self.add(*mobjects)
            self.camera.add_fixed_orientation_mobjects(*mobjects, **kwargs)

    def add_fixed_in_frame_mobjects(self, *mobjects: Mobject):
        """
        This method is used to prevent the rotation and movement
        of mobjects as the camera moves around. The mobject is
        essentially overlaid, and is not impacted by the camera's
        movement in any way.

        Parameters
        ----------
        *mobjects
            The Mobjects whose orientation must be fixed.
        """
        if isinstance(self.camera, ThreeDCamera):
            self.add(*mobjects)
            self.camera.add_fixed_in_frame_mobjects(*mobjects)

    def remove_fixed_orientation_mobjects(self, *mobjects: Mobject):
        """
        This method "unfixes" the orientation of the mobjects
        passed, meaning they will no longer be at the same angle
        relative to the camera. This only makes sense if the
        mobject was passed through add_fixed_orientation_mobjects first.

        Parameters
        ----------
        *mobjects
            The Mobjects whose orientation must be unfixed.
        """
        if isinstance(self.camera, ThreeDCamera):
            self.camera.remove_fixed_orientation_mobjects(*mobjects)

    def remove_fixed_in_frame_mobjects(self, *mobjects: Mobject):
        """
         This method undoes what add_fixed_in_frame_mobjects does.
         It allows the mobject to be affected by the movement of
         the camera.

        Parameters
        ----------
        *mobjects
            The Mobjects whose position and orientation must be unfixed.
        """
        if isinstance(self.camera, ThreeDCamera):
            self.camera.remove_fixed_in_frame_mobjects(*mobjects)

    ##
    def set_to_default_angled_camera_orientation(self, **kwargs):
        """
        This method sets the default_angled_camera_orientation to the
        keyword arguments passed, and sets the camera to that orientation.

        Parameters
        ----------
        **kwargs
            Some recognised kwargs are phi, theta, focal_distance, gamma,
            which have the same meaning as the parameters in set_camera_orientation.
        """
        config = dict(
            self.default_angled_camera_orientation_kwargs,
        )
        config.update(kwargs)
        self.set_camera_orientation(**config)
