"""A camera able to move through a scene.

.. SEEALSO::

    :mod:`.moving_camera_scene`

"""

from __future__ import annotations

__all__ = ["MovingCamera"]

import numpy as np

from .. import config
from ..camera.camera import Camera
from ..constants import DOWN, LEFT, RIGHT, UP
from ..mobject.frame import ScreenRectangle
from ..mobject.mobject import Mobject
from ..utils.color import WHITE


class MovingCamera(Camera):
    """
    Stays in line with the height, width and position of it's 'frame', which is a Rectangle

    .. SEEALSO::

        :class:`.MovingCameraScene`

    """

    def __init__(
        self,
        frame=None,
        fixed_dimension=0,  # width
        default_frame_stroke_color=WHITE,
        default_frame_stroke_width=0,
        **kwargs,
    ):
        """
        Frame is a Mobject, (should almost certainly be a rectangle)
        determining which region of space the camera displays
        """
        self.fixed_dimension = fixed_dimension
        self.default_frame_stroke_color = default_frame_stroke_color
        self.default_frame_stroke_width = default_frame_stroke_width
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
    def frame_height(self):
        """Returns the height of the frame.

        Returns
        -------
        float
            The height of the frame.
        """
        return self.frame.height

    @property
    def frame_width(self):
        """Returns the width of the frame

        Returns
        -------
        float
            The width of the frame.
        """
        return self.frame.width

    @property
    def frame_center(self):
        """Returns the centerpoint of the frame in cartesian coordinates.

        Returns
        -------
        np.array
            The cartesian coordinates of the center of the frame.
        """
        return self.frame.get_center()

    @frame_height.setter
    def frame_height(self, frame_height: float):
        """Sets the height of the frame in MUnits.

        Parameters
        ----------
        frame_height
            The new frame_height.
        """
        self.frame.stretch_to_fit_height(frame_height)

    @frame_width.setter
    def frame_width(self, frame_width: float):
        """Sets the width of the frame in MUnits.

        Parameters
        ----------
        frame_width
            The new frame_width.
        """
        self.frame.stretch_to_fit_width(frame_width)

    @frame_center.setter
    def frame_center(self, frame_center: np.ndarray | list | tuple | Mobject):
        """Sets the centerpoint of the frame.

        Parameters
        ----------
        frame_center
            The point to which the frame must be moved.
            If is of type mobject, the frame will be moved to
            the center of that mobject.
        """
        self.frame.move_to(frame_center)

    def capture_mobjects(self, mobjects, **kwargs):
        # self.reset_frame_center()
        # self.realign_frame_shape()
        super().capture_mobjects(mobjects, **kwargs)

    # Since the frame can be moving around, the cairo
    # context used for updating should be regenerated
    # at each frame.  So no caching.
    def get_cached_cairo_context(self, pixel_array):
        """
        Since the frame can be moving around, the cairo
        context used for updating should be regenerated
        at each frame.  So no caching.
        """
        return None

    def cache_cairo_context(self, pixel_array, ctx):
        """
        Since the frame can be moving around, the cairo
        context used for updating should be regenerated
        at each frame.  So no caching.
        """
        pass

    # def reset_frame_center(self):
    #     self.frame_center = self.frame.get_center()

    # def realign_frame_shape(self):
    #     height, width = self.frame_shape
    #     if self.fixed_dimension == 0:
    #         self.frame_shape = (height, self.frame.width
    #     else:
    #         self.frame_shape = (self.frame.height, width)
    #     self.resize_frame_shape(fixed_dimension=self.fixed_dimension)

    def get_mobjects_indicating_movement(self):
        """
        Returns all mobjects whose movement implies that the camera
        should think of all other mobjects on the screen as moving

        Returns
        -------
        list
        """
        return [self.frame]

    def auto_zoom(
        self,
        mobjects: list[Mobject],
        margin: float = 0,
        only_mobjects_in_frame: bool = False,
        animate: bool = True,
    ):
        """Zooms on to a given array of mobjects (or a singular mobject)
        and automatically resizes to frame all the mobjects.

        .. NOTE::

            This method only works when 2D-objects in the XY-plane are considered, it
            will not work correctly when the camera has been rotated.

        Parameters
        ----------
        mobjects
            The mobject or array of mobjects that the camera will focus on.

        margin
            The width of the margin that is added to the frame (optional, 0 by default).

        only_mobjects_in_frame
            If set to ``True``, only allows focusing on mobjects that are already in frame.

        animate
            If set to ``False``, applies the changes instead of returning the corresponding animation

        Returns
        -------
        Union[_AnimationBuilder, ScreenRectangle]
            _AnimationBuilder that zooms the camera view to a given list of mobjects
            or ScreenRectangle with position and size updated to zoomed position.

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
