"""A scene suitable for rendering three-dimensional objects and animations."""

from __future__ import annotations

__all__ = ["ThreeDScene", "SpecialThreeDScene"]


import warnings
from typing import Iterable, Sequence

import numpy as np

from manim.mobject.geometry.line import Line
from manim.mobject.graphing.coordinate_systems import ThreeDAxes
from manim.mobject.opengl.opengl_mobject import OpenGLMobject
from manim.mobject.three_d.three_dimensions import Sphere
from manim.mobject.value_tracker import ValueTracker

from .. import config
from ..animation.animation import Animation
from ..animation.transform import Transform
from ..camera.three_d_camera import ThreeDCamera
from ..constants import DEGREES, RendererType
from ..mobject.mobject import Mobject
from ..mobject.types.vectorized_mobject import VectorizedPoint, VGroup
from ..renderer.opengl_renderer import OpenGLCamera
from ..scene.scene import Scene
from ..utils.config_ops import merge_dicts_recursively


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
            self.renderer.camera.set_phi(phi)
        if theta is not None:
            self.renderer.camera.set_theta(theta)
        if focal_distance is not None:
            self.renderer.camera.set_focal_distance(focal_distance)
        if gamma is not None:
            self.renderer.camera.set_gamma(gamma)
        if zoom is not None:
            self.renderer.camera.set_zoom(zoom)
        if frame_center is not None:
            self.renderer.camera._frame_center.move_to(frame_center)

    def begin_ambient_camera_rotation(self, rate: float = 0.02, about: str = "theta"):
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
            if config.renderer == RendererType.CAIRO:
                trackers = {
                    "theta": self.camera.theta_tracker,
                    "phi": self.camera.phi_tracker,
                    "gamma": self.camera.gamma_tracker,
                }
                x: ValueTracker = trackers[about]
                x.add_updater(lambda m, dt: x.increment_value(rate * dt))
                self.add(x)
            elif config.renderer == RendererType.OPENGL:
                cam: OpenGLCamera = self.camera
                methods = {
                    "theta": cam.increment_theta,
                    "phi": cam.increment_phi,
                    "gamma": cam.increment_gamma,
                }
                cam.add_updater(lambda m, dt: methods[about](rate * dt))
                self.add(self.camera)
        except Exception:
            raise ValueError("Invalid ambient rotation angle.")

    def stop_ambient_camera_rotation(self, about="theta"):
        """
        This method stops all ambient camera rotation.
        """
        about: str = about.lower()
        try:
            if config.renderer == RendererType.CAIRO:
                trackers = {
                    "theta": self.camera.theta_tracker,
                    "phi": self.camera.phi_tracker,
                    "gamma": self.camera.gamma_tracker,
                }
                x: ValueTracker = trackers[about]
                x.clear_updaters()
                self.remove(x)
            elif config.renderer == RendererType.OPENGL:
                self.camera.clear_updaters()
        except Exception:
            raise ValueError("Invalid ambient rotation angle.")

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
            origin_theta = self.renderer.camera.theta_tracker.get_value()
        if origin_phi is None:
            origin_phi = self.renderer.camera.phi_tracker.get_value()

        val_tracker_theta = ValueTracker(0)

        def update_theta(m, dt):
            val_tracker_theta.increment_value(dt * rate)
            val_for_left_right = 0.2 * np.sin(val_tracker_theta.get_value())
            return m.set_value(origin_theta + val_for_left_right)

        self.renderer.camera.theta_tracker.add_updater(update_theta)
        self.add(self.renderer.camera.theta_tracker)

        val_tracker_phi = ValueTracker(0)

        def update_phi(m, dt):
            val_tracker_phi.increment_value(dt * rate)
            val_for_up_down = 0.1 * np.cos(val_tracker_phi.get_value()) - 0.1
            return m.set_value(origin_phi + val_for_up_down)

        self.renderer.camera.phi_tracker.add_updater(update_phi)
        self.add(self.renderer.camera.phi_tracker)

    def stop_3dillusion_camera_rotation(self):
        """
        This method stops all illusion camera rotations.
        """
        self.renderer.camera.theta_tracker.clear_updaters()
        self.remove(self.renderer.camera.theta_tracker)
        self.renderer.camera.phi_tracker.clear_updaters()
        self.remove(self.renderer.camera.phi_tracker)

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

        if config.renderer == RendererType.CAIRO:
            self.camera: ThreeDCamera
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
                anims.append(self.camera._frame_center.animate.move_to(frame_center))
        elif config.renderer == RendererType.OPENGL:
            cam: OpenGLCamera = self.camera
            cam2 = cam.copy()
            methods = {
                "theta": cam2.set_theta,
                "phi": cam2.set_phi,
                "gamma": cam2.set_gamma,
                "zoom": cam2.scale,
                "frame_center": cam2.move_to,
            }
            if frame_center is not None:
                if isinstance(frame_center, OpenGLMobject):
                    frame_center = frame_center.get_center()
                frame_center = list(frame_center)

            for value, method in [
                [theta, "theta"],
                [phi, "phi"],
                [gamma, "gamma"],
                [
                    config.frame_height / (zoom * cam.height)
                    if zoom is not None
                    else None,
                    "zoom",
                ],
                [frame_center, "frame_center"],
            ]:
                if value is not None:
                    methods[method](value)

            if focal_distance is not None:
                warnings.warn(
                    "focal distance of OpenGLCamera can not be adjusted.",
                    stacklevel=2,
                )

            anims += [Transform(cam, cam2)]

        self.play(*anims + added_anims, **kwargs)

        # These lines are added to improve performance. If manim thinks that frame_center is moving,
        # it is required to redraw every object. These lines remove frame_center from the Scene once
        # its animation is done, ensuring that manim does not think that it is moving. Since the
        # frame_center is never actually drawn, this shouldn't break anything.
        if frame_center is not None and config.renderer == RendererType.CAIRO:
            self.remove(self.camera._frame_center)

    def get_moving_mobjects(self, *animations: Animation):
        """
        This method returns a list of all of the Mobjects in the Scene that
        are moving, that are also in the animations passed.

        Parameters
        ----------
        *animations
            The animations whose mobjects will be checked.
        """
        moving_mobjects = super().get_moving_mobjects(*animations)
        camera_mobjects = self.renderer.camera.get_value_trackers() + [
            self.renderer.camera._frame_center,
        ]
        if any([cm in moving_mobjects for cm in camera_mobjects]):
            return self.mobjects
        return moving_mobjects

    def add_fixed_orientation_mobjects(self, *mobjects: Mobject, **kwargs):
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
        if config.renderer == RendererType.CAIRO:
            self.add(*mobjects)
            self.renderer.camera.add_fixed_orientation_mobjects(*mobjects, **kwargs)
        elif config.renderer == RendererType.OPENGL:
            for mob in mobjects:
                mob: OpenGLMobject
                mob.fix_orientation()
                self.add(mob)

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
        if config.renderer == RendererType.CAIRO:
            self.add(*mobjects)
            self.camera: ThreeDCamera
            self.camera.add_fixed_in_frame_mobjects(*mobjects)
        elif config.renderer == RendererType.OPENGL:
            for mob in mobjects:
                mob: OpenGLMobject
                mob.fix_in_frame()
                self.add(mob)

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
        if config.renderer == RendererType.CAIRO:
            self.renderer.camera.remove_fixed_orientation_mobjects(*mobjects)
        elif config.renderer == RendererType.OPENGL:
            for mob in mobjects:
                mob: OpenGLMobject
                mob.unfix_orientation()
                self.remove(mob)

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
        if config.renderer == RendererType.CAIRO:
            self.renderer.camera.remove_fixed_in_frame_mobjects(*mobjects)
        elif config.renderer == RendererType.OPENGL:
            for mob in mobjects:
                mob: OpenGLMobject
                mob.unfix_from_frame()
                self.remove(mob)

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
            self.default_camera_orientation_kwargs,
        )  # Where doe this come from?
        config.update(kwargs)
        self.set_camera_orientation(**config)


class SpecialThreeDScene(ThreeDScene):
    """An extension of :class:`ThreeDScene` with more settings.

    It has some extra configuration for axes, spheres,
    and an override for low quality rendering. Further key differences
    are:

    * The camera shades applicable 3DMobjects by default,
      except if rendering in low quality.
    * Some default params for Spheres and Axes have been added.

    """

    def __init__(
        self,
        cut_axes_at_radius=True,
        camera_config={"should_apply_shading": True, "exponential_projection": True},
        three_d_axes_config={
            "num_axis_pieces": 1,
            "axis_config": {
                "unit_size": 2,
                "tick_frequency": 1,
                "numbers_with_elongated_ticks": [0, 1, 2],
                "stroke_width": 2,
            },
        },
        sphere_config={"radius": 2, "resolution": (24, 48)},
        default_angled_camera_position={
            "phi": 70 * DEGREES,
            "theta": -110 * DEGREES,
        },
        # When scene is extracted with -l flag, this
        # configuration will override the above configuration.
        low_quality_config={
            "camera_config": {"should_apply_shading": False},
            "three_d_axes_config": {"num_axis_pieces": 1},
            "sphere_config": {"resolution": (12, 24)},
        },
        **kwargs,
    ):
        self.cut_axes_at_radius = cut_axes_at_radius
        self.camera_config = camera_config
        self.three_d_axes_config = three_d_axes_config
        self.sphere_config = sphere_config
        self.default_angled_camera_position = default_angled_camera_position
        self.low_quality_config = low_quality_config
        if self.renderer.camera_config["pixel_width"] == config["pixel_width"]:
            _config = {}
        else:
            _config = self.low_quality_config
        _config = merge_dicts_recursively(_config, kwargs)
        super().__init__(**_config)

    def get_axes(self):
        """Return a set of 3D axes.

        Returns
        -------
        :class:`.ThreeDAxes`
            A set of 3D axes.
        """
        axes = ThreeDAxes(**self.three_d_axes_config)
        for axis in axes:
            if self.cut_axes_at_radius:
                p0 = axis.get_start()
                p1 = axis.number_to_point(-1)
                p2 = axis.number_to_point(1)
                p3 = axis.get_end()
                new_pieces = VGroup(Line(p0, p1), Line(p1, p2), Line(p2, p3))
                for piece in new_pieces:
                    piece.shade_in_3d = True
                new_pieces.match_style(axis.pieces)
                axis.pieces.submobjects = new_pieces.submobjects
            for tick in axis.tick_marks:
                tick.add(VectorizedPoint(1.5 * tick.get_center()))
        return axes

    def get_sphere(self, **kwargs):
        """
        Returns a sphere with the passed keyword arguments as properties.

        Parameters
        ----------
        **kwargs
            Any valid parameter of :class:`~.Sphere` or :class:`~.Surface`.

        Returns
        -------
        :class:`~.Sphere`
            The sphere object.
        """
        config = merge_dicts_recursively(self.sphere_config, kwargs)
        return Sphere(**config)

    def get_default_camera_position(self):
        """
        Returns the default_angled_camera position.

        Returns
        -------
        dict
            Dictionary of phi, theta, focal_distance, and gamma.
        """
        return self.default_angled_camera_position

    def set_camera_to_default_position(self):
        """
        Sets the camera to its default position.
        """
        self.set_camera_orientation(**self.default_angled_camera_position)
