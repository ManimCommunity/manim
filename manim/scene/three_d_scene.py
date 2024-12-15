"""A scene suitable for rendering three-dimensional objects and animations."""

from __future__ import annotations

__all__ = ["ThreeDScene", "SpecialThreeDScene"]


from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from manim._config import config
from manim.animation.animation import Animation
from manim.mobject.geometry.line import Line
from manim.mobject.graphing.coordinate_systems import ThreeDAxes
from manim.mobject.opengl.opengl_mobject import OpenGLMobject
from manim.mobject.opengl.opengl_vectorized_mobject import (
    OpenGLVectorizedPoint,
    OpenGLVGroup,
)
from manim.mobject.three_d.three_dimensions import Sphere
from manim.scene.scene import Scene
from manim.utils.config_ops import merge_dicts_recursively
from manim.utils.deprecation import deprecated

if TYPE_CHECKING:
    pass


class ThreeDScene(Scene):
    """A :class:`Scene` with special configurations and properties that make it
    suitable for 3D scenes.
    """

    @deprecated(
        replacement="Camera.set_orientation",
        message="Use self.camera.set_orientation() instead.",
    )
    def set_camera_orientation(
        self,
        theta: float | None = None,
        phi: float | None = None,
        gamma: float | None = None,
        zoom: float | None = None,
        focal_distance: float | None = None,
        frame_center: OpenGLMobject | Sequence[float] | None = None,
    ) -> None:
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
        """
        self.camera.set_orientation(
            theta, phi, gamma, zoom, focal_distance, frame_center
        )

    @deprecated(
        replacement="Camera.begin_ambient_rotation",
        message="Use self.camera.begin_ambient_rotation() followed by self.add(self.camera) instead.",
    )
    def begin_ambient_camera_rotation(
        self, rate: float = 0.02, about: str = "theta"
    ) -> None:
        """Apply an updater to rotate the camera on every frame by modifying
        one of three Euler angles: "theta" (rotate about the Z axis), "phi"
        (modify the angle between the camera and the Z axis) or "gamma" (rotate
        the camera in its position while it's looking at the same point).

        Parameters
        ----------
        rate
            The rate at which the camera should rotate for the specified
            angle. A positive rate means counterclockwise rotation, and a
            negative rate means clockwise rotation.
        about
            The Euler angle to modify, which can be "theta", "phi" or "gamma".
            Defaults to "theta".
        """
        self.camera.begin_ambient_rotation(rate, about)
        self.add(self.camera)

    @deprecated(
        replacement="Camera.stop_ambient_rotation",
        message="Use self.camera.stop_ambient_rotation() instead.",
    )
    def stop_ambient_camera_rotation(self, about: str = "theta") -> None:
        """Stop ambient camera rotation on the specified angle. If there's a
        corresponding ambient rotation updater applied on the camera, remove
        it.

        Parameters
        ----------
        about
            The Euler angle for which the rotation should stop. This angle can
            be "theta", "phi" or "gamma". Defaults to "theta".
        """
        self.camera.stop_ambient_rotation(about)

    @deprecated(
        replacement="Camera.begin_precession",
        message="Use self.camera.begin_precession() followed by self.add(self.camera) instead.",
    )
    def begin_camera_precession(
        self,
        rate: float = 1.0,
        radius: float = 0.2,
        origin_phi: float | None = None,
        origin_theta: float | None = None,
    ) -> None:
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
        """
        self.camera.begin_precession(rate, radius, origin_phi, origin_theta)
        self.add(self.camera)

    @deprecated(
        replacement="Camera.stop_precession",
        message="Use self.camera.stop_precession() instead.",
    )
    def stop_camera_precession(self):
        """Remove the precession camera updater, if any."""
        self.camera.stop_precession()

    @deprecated(
        replacement="Camera.begin_precession",
        message="Use self.camera.begin_precession() followed by self.add(self.camera) instead.",
    )
    def begin_3dillusion_camera_rotation(
        self,
        rate: float = 1.0,
        radius: float = 0.2,
        origin_phi: float | None = None,
        origin_theta: float | None = None,
    ) -> None:
        """Alias of :meth:`begin_camera_precession`."""
        self.begin_camera_precession(rate, radius, origin_phi, origin_theta)

    @deprecated(
        replacement="Camera.stop_precession",
        message="Use self.camera.stop_precession() instead.",
    )
    def stop_3dillusion_camera_rotation(self):
        """Alias of :meth:`stop_camera_precession`."""
        self.stop_camera_precession()

    def move_camera(
        self,
        theta: float | None = None,
        phi: float | None = None,
        gamma: float | None = None,
        zoom: float | None = None,
        focal_distance: float | None = None,
        frame_center: OpenGLMobject | Sequence[float] | None = None,
        added_anims: Iterable[Animation] = [],
        **kwargs: Any,
    ) -> None:
        """Animate the movement of the camera to the given spherical coordinates.

        Parameters
        ----------
        theta
            The azimuthal angle i.e the angle that spins the camera around the Z_AXIS.
        phi
            The polar angle i.e the angle between Z_AXIS and Camera through ORIGIN in radians.
        gamma
            The rotation of the camera about the vector from the ORIGIN to the Camera.
        focal_distance
            The radial focal_distance between ORIGIN and Camera.
        zoom
            The zoom factor of the camera.
        frame_center
            The new center of the camera frame in cartesian coordinates.
        added_anims
            Any other animations to be played at the same time.
        """
        animation = self.camera.animate.set_orientation(
            theta, phi, gamma, zoom, focal_distance, frame_center
        )

        self.play(animation, *added_anims, **kwargs)

    def get_moving_mobjects(self, *animations: Animation) -> list[OpenGLMobject]:
        """
        This method returns a list of all of the Mobjects in the Scene that
        are moving, that are also in the animations passed.

        Parameters
        ----------
        *animations
            The animations whose mobjects will be checked.
        """
        moving_mobjects = super().get_moving_mobjects(*animations)
        camera_mobjects = self.camera.get_value_trackers()
        if any(cm in moving_mobjects for cm in camera_mobjects):
            return self.mobjects
        return moving_mobjects

    def add_fixed_orientation_mobjects(self, *mobjects: OpenGLMobject) -> None:
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
        for mob in mobjects:
            mob.fix_orientation()
            self.add(mob)

    def add_fixed_in_frame_mobjects(self, *mobjects: OpenGLMobject) -> None:
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
        for mob in mobjects:
            mob.fix_in_frame()
            self.add(mob)

    def remove_fixed_orientation_mobjects(self, *mobjects: OpenGLMobject) -> None:
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
        for mob in mobjects:
            mob.unfix_orientation()
            self.remove(mob)

    def remove_fixed_in_frame_mobjects(self, *mobjects: OpenGLMobject) -> None:
        """
         This method undoes what add_fixed_in_frame_mobjects does.
         It allows the mobject to be affected by the movement of
         the camera.

        Parameters
        ----------
        *mobjects
            The Mobjects whose position and orientation must be unfixed.
        """
        for mob in mobjects:
            mob.unfix_from_frame()
            self.remove(mob)


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
        cut_axes_at_radius: bool = True,
        camera_config: dict = {
            "should_apply_shading": True,
            "exponential_projection": True,
        },
        three_d_axes_config: dict = {
            "num_axis_pieces": 1,
            "axis_config": {
                "unit_size": 2,
                "tick_frequency": 1,
                "numbers_with_elongated_ticks": [0, 1, 2],
                "stroke_width": 2,
            },
        },
        sphere_config: dict = {"radius": 2, "resolution": (24, 48)},
        # When scene is extracted with -l flag, this
        # configuration will override the above configuration.
        low_quality_config: dict = {
            "camera_config": {"should_apply_shading": False},
            "three_d_axes_config": {"num_axis_pieces": 1},
            "sphere_config": {"resolution": (12, 24)},
        },
        **kwargs: Any,
    ) -> None:
        self.cut_axes_at_radius = cut_axes_at_radius
        self.camera_config = camera_config
        self.three_d_axes_config = three_d_axes_config
        self.sphere_config = sphere_config
        self.low_quality_config = low_quality_config
        if self.manager.renderer.camera_config["pixel_width"] == config["pixel_width"]:
            _config = {}
        else:
            _config = self.low_quality_config
        _config = merge_dicts_recursively(_config, kwargs)
        super().__init__(**_config)

    def get_axes(self) -> ThreeDAxes:
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
                new_pieces = OpenGLVGroup(Line(p0, p1), Line(p1, p2), Line(p2, p3))
                for piece in new_pieces:
                    piece.shade_in_3d = True
                new_pieces.match_style(axis.pieces)
                axis.pieces.submobjects = new_pieces.submobjects
                axis.pieces.note_changed_family()
            for tick in axis.tick_marks:
                tick.add(OpenGLVectorizedPoint(1.5 * tick.get_center()))
        return axes

    def get_sphere(self, **kwargs: Any) -> Sphere:
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
