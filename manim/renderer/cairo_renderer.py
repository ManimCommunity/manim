import typing

import numpy as np

from .. import config
from ..camera.camera import Camera
from ..mobject.mobject import Mobject
from ..utils.iterables import list_update
from .renderer import Renderer


class CairoRenderer(Renderer):
    """A renderer using Cairo.

    num_plays : Number of play() functions in the scene.
    time: time elapsed since initialisation of scene.
    """

    def __init__(self, camera_class=None, skip_animations=False, **kwargs):
        # All of the following are set to EITHER the value passed via kwargs,
        # OR the value stored in the global config dict at the time of
        # _instance construction_.
        super().__init__()
        camera_cls = camera_class if camera_class is not None else Camera
        self.camera = camera_cls()
        self._original_skipping_status = skip_animations
        self.skip_animations = skip_animations
        self.animations_hashes = []
        self.num_plays = 0
        self.time = 0
        self.static_image = None

    def init_scene(self):
        pass

    def before_animation(self):
        pass

    def after_animation(self):
        pass

    def before_render(self):
        pass

    def after_render(self):
        pass

    def has_interaction(self):
        return False

    def can_handle_static_wait(self):
        return True

    def update_frame(  # TODO Description in Docstring
        self,
        moving_mobjects,
        skip_animations=False,
        include_submobjects=True,
        ignore_skipping=False,
        mobjects=None,
        meshes=None,
        file_writer=None,
        foreground_mobjects=None,
        **kwargs,
    ):
        """Update the frame.

        Parameters
        ----------
        mobjects: list, optional
            list of mobjects

        background: np.ndarray, optional
            Pixel Array for Background.

        include_submobjects: bool, optional

        ignore_skipping : bool, optional

        **kwargs

        """
        if skip_animations and not ignore_skipping:
            return
        if not moving_mobjects:
            moving_mobjects = list_update(
                mobjects,
                foreground_mobjects,
            )
        if self.static_image is not None:
            self.camera.set_frame_to_background(self.static_image)
        else:
            self.camera.reset()

        kwargs["include_submobjects"] = include_submobjects
        self.camera.capture_mobjects(moving_mobjects, **kwargs)

    def render(
        self,
        time,
        moving_mobjects,
        skip_animations=False,
        mobjects=None,
        meshes=None,
        file_writer=None,
        foreground_mobjects=None,
    ):
        self.update_frame(
            moving_mobjects,
            skip_animations == skip_animations,
            mobjects=mobjects,
            foreground_mobjects=foreground_mobjects,
        )
        self.add_frame(self.get_frame(), file_writer, skip_animations=skip_animations)

    def get_frame(self):
        """
        Gets the current frame as NumPy array.

        Returns
        -------
        np.array
            NumPy array of pixel values of each pixel in screen.
            The shape of the array is height x width x 3
        """
        return np.array(self.camera.pixel_array)

    def add_frame(self, frame, file_writer, num_frames=1, skip_animations=False):
        """
        Adds a frame to the video_file_stream

        Parameters
        ----------
        frame : numpy.ndarray
            The frame to add, as a pixel array.
        num_frames: int
            The number of times to add frame.
        """
        dt = 1 / self.camera.frame_rate
        if skip_animations:
            return
        self.time += num_frames * dt
        for _ in range(num_frames):
            file_writer.write_frame(frame)

    def freeze_current_frame(self, duration: float, file_writer, skip_animations=False):
        """Adds a static frame to the movie for a given duration. The static frame is the current frame.

        Parameters
        ----------
        duration : float
            [description]
        """
        dt = 1 / self.camera.frame_rate
        self.add_frame(
            self.get_frame(),
            file_writer,
            skip_animations=skip_animations,
            num_frames=int(duration / dt),
        )

    def show_frame(self):
        """
        Opens the current frame in the Default Image Viewer
        of your system.
        """
        self.update_frame(ignore_skipping=True)
        self.camera.get_image().show()

    def save_static_frame_data(
        self,
        static_mobjects: typing.Iterable[Mobject],
        mobjects=None,
        foreground_mobjects=None,
    ) -> typing.Iterable[Mobject]:
        """Compute and save the static frame, that will be reused at each frame to avoid to unecesseraly computer
        static mobjects.

        Parameters
        ----------
        scene : Scene
            The scene played.
        static_mobjects : typing.Iterable[Mobject]
            Static mobjects of the scene. If None, self.static_image is set to None

        Returns
        -------
        typing.Iterable[Mobject]
            the static image computed.
        """
        if not static_mobjects:
            self.static_image = None
            return
        self.update_frame(
            static_mobjects, mobjects=mobjects, foreground_mobjects=foreground_mobjects
        )
        self.static_image = self.get_frame()
        return self.static_image

    def should_save_last_frame(self, num_plays):
        return config["save_last_frame"] or num_plays == 0
