from ..renderer.stream_renderer import StreamCairoRenderer
from .scene import Scene
from ..camera.camera import Camera
from ..mobject.frame import FullScreenRectangle as Frame


class Stream:
    """This class is really intended for inheritance of the style:

    >>> class Streamer(Stream, Scene): # doctest: +SKIP
    ...     pass
    ...
    >>>

    This order is paramount. This :class:`Stream` class does the switch to
    the specialized renderer, which uses the :class:`StreamFileWriter` to
    handle specialized streaming services. That explains the calls to super,
    which dig through the MRO of a class instead of using just a single
    implementation contained in Scene. Okay, the references in super probably
    point to things in the :class:`Scene` class only, but it's already happened so...

    PS: You can probably already tell using this class on its own will bring you
    errors.

    PPS: Check this bonus:

    >>> class Streamer(Stream, MovingCameraScene, LinearTransformationScene): # doctest: +SKIP
    ...     pass
    ...
    >>>

    """

    def __init__(self, **kwargs):
        # TODO: Someday, when this is accepted into the community, work on camera
        # qualities that can be set from this initialization
        camera = self.get_camera_class()
        renderer = StreamCairoRenderer(camera_class=camera)
        super().__init__(renderer=renderer, **kwargs)
        self.add(Frame())

    @classmethod
    def get_camera_class(cls):
        """Desperately searches for camera classes in CONFIG dictionaries
        in the class hierachy.

        Returns:
            Camera: Camera object intended to be used by the StreamingScene
        """
        for scene in cls.mro():
            CONFIG = getattr(scene, "CONFIG", {})
            if "camera_class" in CONFIG:
                return CONFIG["camera_class"]
        return Camera  # This really shouldn't happen but defaults

    def render(self):
        """This is a recent development I landed on.

        >>> from example_scenes.basic import OpeningManimExample  # doctest: +SKIP
        >>> manim = get_streamer(OpeningManimExample)             # doctest: +SKIP
        >>> manim.render()                                        # doctest: +SKIP

        This should stream a complete rendering of the Scene to the URL specified.
        Hence I clear everything after it's finished for more use. Or something like that.
        """
        super().render()
        self.clear()

    def show_frame(self):
        """
        Opens the current frame in the Default Image Viewer
        of your system.
        """
        self.renderer.update_frame(self, ignore_skipping=True)
        self.renderer.camera.get_image().show()


def get_streamer(*scenes):
    """Creates an instance of a class that has streaming services.

    Optional arguments:
        scenes: Scene classes whose methods can be used in the resulting
        instance, such as zooming in and arbitrary method constructions.
        Defaults to just Scene

    Returns:
        StreamingScene: It's a Scene that Streams. Name deconstruction.
    """
    bases = (Stream,) + (scenes or (Scene,))
    cls = type("StreamingScene", bases, {})
    # This class doesn't really need a name, but we can go
    # generic for this one
    return cls()
