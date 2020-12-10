from .. import config
from ..mobject.frame import FullScreenRectangle as Frame
from ..renderer.stream_renderer import StreamCairoRenderer
from .moving_camera_scene import MovingCameraScene
from .scene import Scene


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

    """

    def __init__(self, **kwargs):
        # TODO: Someday, when this is accepted into the community, work on camera
        # qualities that can be set from this initialization
        super().__init__(**kwargs)
        # I let the thing initialize the other renderer, and kick it out with this line
        self.renderer = StreamCairoRenderer(camera_class=self.camera_class)
        self.renderer.init()
        # To identify the frame in a black background
        self.add(Frame())
        # TODO: What happens when setup actually has play arguments?
        self.setup()

    def show_frame(self):
        """
        Opens the current frame in the Default Image Viewer
        of your system.
        """
        self.renderer.update_frame(self, ignore_skipping=True)
        self.renderer.camera.get_image().show()


def get_streamer(*scene):
    """Creates an instance of a class that has streaming services.

    Optional arguments:
        scene: The scene whose methods can be used in the resulting
        instance, such as zooming in and arbitrary method constructions.
        Defaults to just Scene

    Returns:
        StreamingScene: It's a Scene that Streams. Name deconstruction.

    Note:
        Using starred positional args makes the tuple of the passed scene
        for me. At the moment I figure multiple inheritance is extremely
        delicate and would only work if the classes after the first merely
        add more methods than override base ones. Then again, multiple
        inheritance of scenes was always a delicate matter in Manim.
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
    Hence I clear everything after it's finished for more use. Or something like that.

    Arguments:
        scene: The scene to be played.
        start: The animation to start with. Default original start point.
        end: The animation to end with. Default original endpoint
             Note: The animations use endpoint-inclusive indexing, meaning (0, 5) would
             play 0 upto 5 inclusive of both.
    """
    manim = get_streamer(scene)
    if start or end:
        original = (config.from_animation_number, config.upto_animation_number)
        config.from_animation_number = start or config.from_animation_number
        config.upto_animation_number = end or config.upto_animation_number
    manim.render()
    # Need to put it back because an end point less than the number of animations
    # in a streamer makes any others ignored
    config.from_animation_number, config.upto_animation_number = original
    manim.clear()
