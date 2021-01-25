from .. import config
from ..mobject.frame import FullScreenRectangle as Frame
from ..renderer.stream_renderer import StreamCairoRenderer
from ..utils.simple_functions import get_parameters
from .scene import Scene

import math


class Stream:
    """Abstract base class.

    This class is really intended for inheritance of the style::
        >>> class Streamer(Stream, Scene): # doctest: +SKIP
        ...     pass
        ...
        >>>

    This order is paramount. This :class:`Stream` class carries out the switch to
    the specialized renderer, which uses :class:`StreamFileWriter` to
    handle specialized streaming services. That explains the calls to ``super``,
    which digs through the MRO of a class instead of using just a single
    implementation contained in Scene.

    .. note::

        This class is not intended to be used on its own and will
        most likely raise errors if done so.
    """

    def __init__(self, **kwargs):
        camera_class = self.mint_camera_class()
        renderer = StreamCairoRenderer(camera_class=camera_class)
        super().__init__(renderer=renderer, **kwargs)
        # To identify the frame in a black background
        self.add(Frame())
        self.setup()

    @classmethod
    def mint_camera_class(cls):
        """A camera class from the scene's inheritance hierarchy.

        Only ``__init__`` methods in :class:`~.Scene` classes and derived classes
        from this have the camera class required for the renderer. This declaration
        for the entire class exists only here, and for that reason it is the only place
        to look.

        Raises
        ------
        AttributeError
            If this lookup fails.
        """

        for obj in cls.mro():
            try:
                parameter = get_parameters(obj.__init__)["camera_class"]
            except KeyError:
                continue
            else:
                return parameter.default
        raise AttributeError("Object does not contain scene protocol")

    def show_frame(self):
        """
        Opens the current frame in the Default Image Viewer
        of your system.
        """
        self.renderer.update_frame(self, ignore_skipping=True)
        self.renderer.camera.get_image().show()


def get_streamer(*scene):
    """
    Parameters
    ----------
    scene
        The scene whose methods can be used in the resulting
        instance, such as zooming in and arbitrary method constructions.
        Defaults to just Scene

    Returns
    -------
    StreamingScene
        A scene suited for streaming.
    """
    bases = (Stream,) + (scene or (Scene,))
    cls = type("StreamingScene", bases, {})
    # This class doesn't really need a name, but we can go
    # generic for this one
    return cls()


def play_scene(scene, start=None, end=None):
    """Every scene has a render method that runs its setup and construct methods.
    Using a streamer from classes with detailed implementation of this may call for
    use of this.

    >>> from example_scenes.basic import OpeningManimExample  # doctest: +SKIP
    >>> manim = get_streamer(OpeningManimExample)             # doctest: +SKIP
    >>> manim.render()                                        # doctest: +SKIP

    This should stream a complete rendering of the Scene to the URL specified.
    Hence the function clears everything after it's finished for more use. Or
    something like that.

    Parameters
    ----------
    scene
        The scene to be played.
    start
        The animation to start with. Default original start point.
    end
        The animation to end with. Default original endpoint

    .. note::
        The animations use endpoint-inclusive indexing, meaning (0, 5) would
        play 0 upto 5 inclusive of both.
    """
    manim = get_streamer(scene)
    config.from_animation_number = start or 0
    config.upto_animation_number = end or math.inf
    manim.render()
    # Need to put it back because an end point less than the number of animations
    # in a streamer makes any others ignored. That's a bug
    config.from_animation_number, config.upto_animation_number = 0, math.inf
    manim.clear()
