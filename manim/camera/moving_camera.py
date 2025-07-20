"""A camera able to move through a scene.

.. SEEALSO::

    :mod:`.moving_camera_scene`

"""

from __future__ import annotations

__all__ = ["MovingCamera"]

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np

from manim import config
from manim.camera.camera import Camera
from manim.constants import DOWN, LEFT, RIGHT, UP
from manim.mobject.frame import ScreenRectangle
from manim.mobject.mobject import Mobject
from manim.utils.color import WHITE

if TYPE_CHECKING:
    import cairo

    from manim.mobject.mobject import _AnimationBuilder
    from manim.typing import Point3D, Point3DLike
    from manim.utils.color import ParsableManimColor


class MovingCamera(Camera):
    """
    Subclass of :class:`~.Camera` equipped with a special attribute :attr:`frame`:
    a :class:`~.ScreenRectangle` delimiting the region displayed
    by the Camera, which follows the frame's position and stays in line with
    its height and width.

    .. SEEALSO::

        :class:`.MovingCameraScene`

    Attributes
    ----------
    frame : :class:`~.ScreenRectangle`
        A :class:`~.ScreenRectangle` which determines the region of space displayed
        by :class:`MovingCamera`.
    fixed_dimension
        Currently unused.
    default_frame_stroke_color
        Default stroke color for the border of :attr:`frame`.
    default_frame_stroke_width
        Default stroke width for the border of :attr:`frame`.

    Parameters
    ----------
    frame
        An optional :class:`~.ScreenRectangle` which determines the region of space
        displayed by :class:`MovingCamera`. If ``None``, a new :class:`~.ScreenRectangle`
        is generated automatically for the camera.
    fixed_dimension
        Currently unused.
    default_frame_stroke_color
        Default stroke color for the border of :attr:`frame`.
    default_frame_stroke_width
        Default stroke width for the border of :attr:`frame`.
    """

    def __init__(
        self,
        frame: ScreenRectangle | None = None,
        fixed_dimension: int = 0,  # width
        default_frame_stroke_color: ParsableManimColor | None = WHITE,
        default_frame_stroke_width: float = 0,
        **kwargs,
    ) -> None:
        self.fixed_dimension = fixed_dimension
        self.default_frame_stroke_color = default_frame_stroke_color
        self.default_frame_stroke_width = default_frame_stroke_width
        """
        Frame is a Mobject (should almost certainly be a rectangle)
        which determines the region of space displayed by the camera.
        """
        if frame is None:
            frame = ScreenRectangle(height=config["frame_height"])
            frame.set_stroke(
                self.default_frame_stroke_color,
                self.default_frame_stroke_width,
            )
        self.frame = frame
        super().__init__(**kwargs)

    # TODO, make these work for a rotated frame
    @property
    def frame_height(self) -> float:
        """Returns the height of :attr:`frame`.

        Returns
        -------
        float
            The height of :attr:`frame`.
        """
        return self.frame.height

    @frame_height.setter
    def frame_height(self, frame_height: float) -> None:
        """Sets the height of :attr:`frame` in MUnits.

        Parameters
        ----------
        frame_height
            The new height for :attr:`frame`.
        """
        self.frame.stretch_to_fit_height(frame_height)

    @property
    def frame_width(self) -> float:
        """Returns the width of :attr:`frame`.

        Returns
        -------
        float
            The width of :attr:`frame`.
        """
        return self.frame.width

    @frame_width.setter
    def frame_width(self, frame_width: float) -> None:
        """Sets the width of :attr:`frame` in MUnits.

        Parameters
        ----------
        frame_width
            The new width for :attr:`frame`.
        """
        self.frame.stretch_to_fit_width(frame_width)

    @property
    def frame_center(self) -> Point3D:
        """Returns the centerpoint of :attr:`frame` in Cartesian coordinates.

        Returns
        -------
        np.array
            The Cartesian coordinates of the center of :attr:`frame`.
        """
        return self.frame.get_center()

    @frame_center.setter
    def frame_center(self, frame_center: Point3DLike | Mobject) -> None:
        """Sets the centerpoint of :attr:`frame`.

        Parameters
        ----------
        frame_center
            Point to which :attr:`frame` must be moved, or another
            :class:`~.Mobject` whose center will be used for :attr:`frame`.
        """
        self.frame.move_to(frame_center)

    # TODO: If the other methods are commented and this only
    # calls super(), this override might as well be deleted
    def capture_mobjects(
        self,
        mobjects: Iterable[Mobject],
        **kwargs: Any,
    ) -> None:
        # self.reset_frame_center()
        # self.realign_frame_shape()
        super().capture_mobjects(mobjects, **kwargs)

    # Since the frame can be moving around, the Cairo
    # context used for updating should be regenerated
    # at each frame. So no caching.
    def get_cached_cairo_context(self, pixel_array: np.ndarray) -> None:
        """
        Since the frame can be moving around, the Cairo
        context used for updating should be regenerated
        at each frame. So no caching.
        """
        return None

    def cache_cairo_context(self, pixel_array: np.ndarray, ctx: cairo.Context) -> None:
        """
        Since the frame can be moving around, the Cairo
        context used for updating should be regenerated
        at each frame. So no caching.
        """
        pass

    # def reset_frame_center(self) -> None:
    #     self.frame_center = self.frame.get_center()

    # def realign_frame_shape(self) -> None:
    #     height, width = self.frame_shape
    #     if self.fixed_dimension == 0:
    #         self.frame_shape = (height, self.frame.width
    #     else:
    #         self.frame_shape = (self.frame.height, width)
    #     self.resize_frame_shape(fixed_dimension=self.fixed_dimension)

    def get_mobjects_indicating_movement(self) -> list[Mobject]:
        """Returns all Mobjects whose movement implies that
        the :class:`MovingCamera` should think of all the other Mobjects
        on the screen as moving.

        Returns
        -------
        List[:class:`Mobject`]
            List of Mobjects indicating movement.
        """
        return [self.frame]

    def auto_zoom(
        self,
        mobjects: Mobject | Iterable[Mobject],
        margin: float = 0,
        only_mobjects_in_frame: bool = False,
        animate: bool = True,
    ) -> _AnimationBuilder | ScreenRectangle:
        """Zooms on to a given array of Mobjects (or a singular Mobject)
        and automatically resizes to frame all the Mobjects.

        .. NOTE::

            This method only works when 2D-objects in the XY-plane are considered, it
            will not work correctly when the camera has been rotated.

        Parameters
        ----------
        mobjects
            The :class:`~.Mobject` or array of Mobjects that the :class:`MovingCamera` will focus on.

        margin
            The width of the margin that is added to the frame (optional, 0 by default).

        only_mobjects_in_frame
            If set to ``True``, only allows focusing on Mobjects that are already in frame.

        animate
            If set to ``False``, applies the changes instead of returning the corresponding animation.

        Returns
        -------
        :class:`~._AnimationBuilder` | :class:`~.ScreenRectangle`
            An :class:`~._AnimationBuilder` that zooms the camera view to a given list of Mobjects,
            or a :class:`~.ScreenRectangle` with its position and size updated to the zoomed position.

        """
        scene_critical_x_left = None
        scene_critical_x_right = None
        scene_critical_y_up = None
        scene_critical_y_down = None

        for m in mobjects:
            if (m == self.frame) or (
                only_mobjects_in_frame and not self.is_in_frame(m)
            ):
                # detected camera frame, should not be used to calculate final position of camera
                continue

            # initialize scene critical points with first mobjects critical points
            if scene_critical_x_left is None:
                scene_critical_x_left = m.get_critical_point(LEFT)[0]
                scene_critical_x_right = m.get_critical_point(RIGHT)[0]
                scene_critical_y_up = m.get_critical_point(UP)[1]
                scene_critical_y_down = m.get_critical_point(DOWN)[1]

            else:
                if m.get_critical_point(LEFT)[0] < scene_critical_x_left:
                    scene_critical_x_left = m.get_critical_point(LEFT)[0]

                if m.get_critical_point(RIGHT)[0] > scene_critical_x_right:
                    scene_critical_x_right = m.get_critical_point(RIGHT)[0]

                if m.get_critical_point(UP)[1] > scene_critical_y_up:
                    scene_critical_y_up = m.get_critical_point(UP)[1]

                if m.get_critical_point(DOWN)[1] < scene_critical_y_down:
                    scene_critical_y_down = m.get_critical_point(DOWN)[1]

        # calculate center x and y
        x = (scene_critical_x_left + scene_critical_x_right) / 2
        y = (scene_critical_y_up + scene_critical_y_down) / 2

        # calculate proposed width and height of zoomed scene
        new_width = abs(scene_critical_x_left - scene_critical_x_right)
        new_height = abs(scene_critical_y_up - scene_critical_y_down)

        m_target = self.frame.animate if animate else self.frame
        # zoom to fit all mobjects along the side that has the largest size
        if new_width / self.frame.width > new_height / self.frame.height:
            return m_target.set_x(x).set_y(y).set(width=new_width + margin)
        else:
            return m_target.set_x(x).set_y(y).set(height=new_height + margin)
