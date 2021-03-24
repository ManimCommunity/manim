import typing
import time
import numpy as np

from manim.utils.hashing import get_hash_from_play_call

from .. import config, logger
from ..camera.camera import Camera
from ..scene.scene_file_writer import SceneFileWriter
from ..utils.exceptions import EndSceneEarlyException
from ..utils.iterables import list_update
from ..mobject.mobject import Mobject


def handle_play_like_call(func):
    """
    This method is used internally to wrap the
    passed function, into a function that
    actually writes to the video stream.
    Simultaneously, it also adds to the number
    of animations played.

    Parameters
    ----------
    func : function
        The play() like function that has to be
        written to the video file stream.

    Returns
    -------
    function
        The play() like function that can now write
        to the video file stream.
    """

    # NOTE : This is only kept for OpenGL renderer.
    # The play logic of the cairo renderer as been refactored and does not need this function anymore.
    # When OpenGL renderer will have a proper testing system,
    # the play logic of the latter has to be refactored in the same way the cairo renderer has been, and thus this
    # method has to be deleted.

    def wrapper(self, scene, *args, **kwargs):
        self.animation_start_time = time.time()
        self.file_writer.begin_animation(not self.skip_animations)
        func(self, scene, *args, **kwargs)
        self.file_writer.end_animation(not self.skip_animations)
        self.num_plays += 1

    return wrapper


class CairoRenderer:
    """A renderer using Cairo.

    num_plays : Number of play() functions in the scene.
    time: time elapsed since initialisation of scene.
    """

    def __init__(self, camera_class=None, skip_animations=False, **kwargs):
        # All of the following are set to EITHER the value passed via kwargs,
        # OR the value stored in the global config dict at the time of
        # _instance construction_.
        self.file_writer = None
        camera_cls = camera_class if camera_class is not None else Camera
        self.camera = camera_cls()
        self._original_skipping_status = skip_animations
        self.skip_animations = skip_animations
        self.animations_hashes = []
        self.num_plays = 0
        self.time = 0
        self.static_image = None

    def init_scene(self, scene):
        self.file_writer = SceneFileWriter(
            self,
            scene.__class__.__name__,
        )

    def play(self, scene, *args, **kwargs):
        # Reset skip_animations to the original state.
        # Needed when rendering only some animations, and skipping others.
        self.skip_animations = self._original_skipping_status
        self.update_skipping_status()

        scene.compile_animation_data(*args, **kwargs)

        # If skip_animations is already True, we can skip all the caching process.
        if not config["disable_caching"] and not self.skip_animations:
            hash_current_animation = get_hash_from_play_call(
                scene, self.camera, scene.animations, scene.mobjects
            )
            if self.file_writer.is_already_cached(hash_current_animation):
                logger.info(
                    f"Animation {self.num_plays} : Using cached data (hash : %(hash_current_animation)s)",
                    {"hash_current_animation": hash_current_animation},
                )
                self.skip_animations = True
        else:
            hash_current_animation = f"uncached_{self.num_plays:05}"

        if self.skip_animations:
            logger.debug(f"Skipping animation {self.num_plays}")
            hash_current_animation = None

        # adding None as a partial movie file will make file_writer ignore the latter.
        self.file_writer.add_partial_movie_file(hash_current_animation)
        self.animations_hashes.append(hash_current_animation)
        logger.debug(
            "List of the first few animation hashes of the scene: %(h)s",
            {"h": str(self.animations_hashes[:5])},
        )

        # Save a static image, to avoid rendering non moving objects.
        self.static_image = self.save_static_frame_data(scene, scene.static_mobjects)

        self.file_writer.begin_animation(not self.skip_animations)
        scene.begin_animations()
        if scene.is_current_animation_frozen_frame():
            self.update_frame(scene)
            # self.duration stands for the total run time of all the animations.
            # In this case, as there is only a wait, it will be the length of the wait.
            self.freeze_current_frame(scene.duration)
        else:
            scene.play_internal()
        self.file_writer.end_animation(not self.skip_animations)

        self.num_plays += 1

    def update_frame(  # TODO Description in Docstring
        self,
        scene,
        mobjects=None,
        include_submobjects=True,
        ignore_skipping=True,
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
        if self.skip_animations and not ignore_skipping:
            return
        if mobjects is None:
            mobjects = list_update(
                scene.mobjects,
                scene.foreground_mobjects,
            )
        if self.static_image is not None:
            self.camera.set_frame_to_background(self.static_image)
        else:
            self.camera.reset()

        kwargs["include_submobjects"] = include_submobjects
        self.camera.capture_mobjects(mobjects, **kwargs)

    def render(self, scene, time, moving_mobjects):
        self.update_frame(scene, moving_mobjects)
        self.add_frame(self.get_frame())

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

    def add_frame(self, frame, num_frames=1):
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
        self.time += num_frames * dt
        if self.skip_animations:
            return
        for _ in range(num_frames):
            self.file_writer.write_frame(frame)

    def freeze_current_frame(self, duration: float):
        """Adds a static frame to the movie for a given duration. The static frame is the current frame.

        Parameters
        ----------
        duration : float
            [description]
        """
        dt = 1 / self.camera.frame_rate
        self.add_frame(
            self.get_frame(),
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
        self, scene, static_mobjects: typing.Iterable[Mobject]
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
        if static_mobjects == None or len(static_mobjects) == 0:
            self.static_image = None
            return
        self.update_frame(scene, mobjects=static_mobjects)
        self.static_image = self.get_frame()
        return self.static_image

    def update_skipping_status(self):
        """
        This method is used internally to check if the current
        animation needs to be skipped or not. It also checks if
        the number of animations that were played correspond to
        the number of animations that need to be played, and
        raises an EndSceneEarlyException if they don't correspond.
        """
        if config["save_last_frame"]:
            self.skip_animations = True
        if config["from_animation_number"]:
            if self.num_plays < config["from_animation_number"]:
                self.skip_animations = True
        if config["upto_animation_number"]:
            if self.num_plays > config["upto_animation_number"]:
                self.skip_animations = True
                raise EndSceneEarlyException()

    def scene_finished(self, scene):
        self.file_writer.finish()
        if config["save_last_frame"]:
            self.update_frame(scene, ignore_skipping=False)
            self.file_writer.save_final_image(self.camera.get_image())
