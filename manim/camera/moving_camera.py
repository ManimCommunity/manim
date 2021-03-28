"""A camera able to move through a scene.

.. SEEALSO::

    :mod:`.moving_camera_scene`

"""

__all__ = ["CameraFrame", "MovingCamera"]


from .. import config
from ..camera.camera import Camera
from ..constants import ORIGIN
from ..mobject.frame import ScreenRectangle
from ..mobject.types.vectorized_mobject import VGroup
from ..utils.color import WHITE
from typing import Optional, Union
from ..mobject.mobject import Mobject
import numpy as np


# TODO, think about how to incorporate perspective
class CameraFrame(VGroup):
    def __init__(self, center: np.ndarray = ORIGIN, **kwargs):
        VGroup.__init__(self, center=center, **kwargs)
        self.width = config["frame_width"]
        self.height = config["frame_height"]


class MovingCamera(Camera):
    """
    Stays in line with the height, width and position of it's 'frame', which is a Rectangle

    .. SEEALSO::

        :class:`.MovingCameraScene`

    """

    def __init__(
        self,
        frame: Optional[Mobject] = None,
        fixed_dimension: int = 0,  # width
        default_frame_stroke_color: str = WHITE,
        default_frame_stroke_width: int = 0,
        **kwargs
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
        Camera.__init__(self, **kwargs)

    # TODO, make these work for a rotated frame
    @property
    def frame_height(self) -> float:
        """Returns the height of the frame.

        Returns
        -------
        Hight
            The height of the frame.
        """
        return self.frame.height

    @property
    def frame_width(self) -> float:
        """Returns the width of the frame

        Returns
        -------
        Width
            The width of the frame.
        """
        return self.frame.width

    @property
    def frame_center(self) -> np.array:
        """Returns the centerpoint of the frame in cartesian coordinates.

        Returns
        -------
        Center
            The cartesian coordinates of the center of the frame.
        """
        return self.frame.get_center()

    @frame_height.setter
    def frame_height(self, frame_height: Union[int, float]):
        """Sets the height of the frame in MUnits.

        Parameters
        ----------
        frame_height
            The new frame_height.
        """
        self.frame.stretch_to_fit_height(frame_height)

    @frame_width.setter
    def frame_width(self, frame_width: Union[int, float]):
        """Sets the width of the frame in MUnits.

        Parameters
        ----------
        frame_width
            The new frame_width.
        """
        self.frame.stretch_to_fit_width(frame_width)

    @frame_center.setter
    def frame_center(self, frame_center: Union[np.array, list, tuple, Mobject]):
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
        Camera.capture_mobjects(self, mobjects, **kwargs)

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

    def get_mobjects_indicating_movement(self):
        """
        Returns all mobjects whose movement implies that the camera
        should think of all other mobjects on the screen as moving

        Returns
        -------
        list
        """
        return [self.frame]
