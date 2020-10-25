"""A scene whose camera can be moved around.

.. SEEALSO::

    :mod:`.moving_camera`

"""

__all__ = ["MovingCameraScene"]

from ..camera.moving_camera import MovingCamera
from ..scene.scene import Scene
from ..utils.iterables import list_update
from ..utils.family import extract_mobject_family_members


class MovingCameraScene(Scene):
    """
    This is a Scene, with special configurations and properties that
    make it suitable for cases where the camera must be moved around.

    .. SEEALSO::

        :class:`.MovingCamera`
    """

    CONFIG = {"camera_class": MovingCamera}

    def setup(self):
        """
        This method is used internally by Manim
        to set up the scene for proper use.
        """
        Scene.setup(self)
        assert isinstance(self.renderer.camera, MovingCamera)
        self.camera_frame = self.renderer.camera.frame
        # Hmm, this currently relies on the fact that MovingCamera
        # willd default to a full-sized frame.  Is that okay?
        return self

    def get_moving_mobjects(self, *animations):
        """
        This method returns a list of all of the Mobjects in the Scene that
        are moving, that are also in the animations passed.

        Parameters
        ----------
        *animations : Animation
            The Animations whose mobjects will be checked.
        """
        moving_mobjects = Scene.get_moving_mobjects(self, *animations)
        all_moving_mobjects = extract_mobject_family_members(moving_mobjects)
        movement_indicators = self.renderer.camera.get_mobjects_indicating_movement()
        for movement_indicator in movement_indicators:
            if movement_indicator in all_moving_mobjects:
                # When one of these is moving, the camera should
                # consider all mobjects to be moving
                return list_update(self.mobjects, moving_mobjects)
        return moving_mobjects
