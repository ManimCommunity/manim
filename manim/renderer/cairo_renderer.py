from __future__ import annotations

import typing

import numpy as np

from manim.utils.hashing import get_hash_from_play_call

from .. import config, logger
from ..camera.camera import Camera
from ..mobject.mobject import Mobject
from ..scene.scene_file_writer import SceneFileWriter
from ..utils.exceptions import EndSceneEarlyException
from ..utils.iterables import list_update

if typing.TYPE_CHECKING:
    import types
    from typing import Any, Iterable

    from manim.animation.animation import Animation
    from manim.scene.scene import Scene

__all__ = ["CairoRenderer"]


class CairoRenderer:
    """A renderer using Cairo.

    num_plays : Number of play() functions in the scene.
    time: time elapsed since initialisation of scene.
    """

    def __init__(
        self,
        file_writer_class=SceneFileWriter,
        camera_class=None,
        skip_animations=False,
        **kwargs,
    ):
        # All of the following are set to EITHER the value passed via kwargs,
        # OR the value stored in the global config dict at the time of
        # _instance construction_.
        self._file_writer_class = file_writer_class
        camera_cls = camera_class if camera_class is not None else Camera
        self.camera = camera_cls()
        self._original_skipping_status = skip_animations
        self.skip_animations = skip_animations
        self.animations_hashes = []
        self.num_plays = 0
        self.time = 0
        self.static_image = None

    def init_scene(self, scene):
        self.file_writer: Any = self._file_writer_class(
            self,
            scene.__class__.__name__,
        )

    def play(
        self,
        scene: Scene,
        *args: Animation | Iterable[Animation] | types.GeneratorType[Animation],
        **kwargs,
    ):
        # Reset skip_animations to the original state.
        # Needed when rendering only some animations, and skipping others.
        self.skip_animations = self._original_skipping_status
        self.update_skipping_status()

        scene.compile_animation_data(*args, **kwargs)

        if self.skip_animations:
            logger.debug(f"Skipping animation {self.num_plays}")
            hash_current_animation = None
            self.time += scene.duration
        else:
            if config["disable_caching"]:
                logger.info("Caching disabled.")
                hash_current_animation = f"uncached_{self.num_plays:05}"
            else:
                hash_current_animation = get_hash_from_play_call(
                    scene,
                    self.camera,
                    scene.animations,
                    scene.mobjects,
                )
                if self.file_writer.is_already_cached(hash_current_animation):
                    logger.info(
                        f"Animation {self.num_plays} : Using cached data (hash : %(hash_current_animation)s)",
                        {"hash_current_animation": hash_current_animation},
                    )
                    self.skip_animations = True
                    self.time += scene.duration
        # adding None as a partial movie file will make file_writer ignore the latter.
        self.file_writer.add_partial_movie_file(hash_current_animation)
        self.animations_hashes.append(hash_current_animation)
        logger.debug(
            "List of the first few animation hashes of the scene: %(h)s",
            {"h": str(self.animations_hashes[:5])},
        )

        self.file_writer.begin_animation(not self.skip_animations)
        scene.begin_animations()

        # Save a static image, to avoid rendering non moving objects.
        self.save_static_frame_data(scene, scene.static_mobjects)

        if scene.is_current_animation_frozen_frame():
            self.update_frame(scene, mobjects=scene.moving_mobjects)
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
        mobjects: typing.Iterable[Mobject] | None = None,
        include_submobjects: bool = True,
        ignore_skipping: bool = True,
        **kwargs,
    ):
        """Update the frame.

        Parameters
        ----------
        scene

        mobjects
            list of mobjects

        include_submobjects

        ignore_skipping

        **kwargs

        """
        if self.skip_animations and not ignore_skipping:
            return
        if not mobjects:
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

    def add_frame(self, frame: np.ndarray, num_frames: int = 1):
        """
        Adds a frame to the video_file_stream

        Parameters
        ----------
        frame
            The frame to add, as a pixel array.
        num_frames
            The number of times to add frame.
        """
        dt = 1 / self.camera.frame_rate
        if self.skip_animations:
            return
        self.time += num_frames * dt
        for _ in range(num_frames):
            self.file_writer.write_frame(frame)

    def freeze_current_frame(self, duration: float):
        """Adds a static frame to the movie for a given duration. The static frame is the current frame.

        Parameters
        ----------
        duration
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
        self,
        scene: Scene,
        static_mobjects: typing.Iterable[Mobject],
    ) -> typing.Iterable[Mobject] | None:
        """Compute and save the static frame, that will be reused at each frame
        to avoid unnecessarily computing static mobjects.

        Parameters
        ----------
        scene
            The scene played.
        static_mobjects
            Static mobjects of the scene. If None, self.static_image is set to None

        Returns
        -------
        typing.Iterable[Mobject]
            The static image computed.
        """
        self.static_image = None
        if not static_mobjects:
            return None
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
        # there is always at least one section -> no out of bounds here
        if self.file_writer.sections[-1].skip_animations:
            self.skip_animations = True
        if config["save_last_frame"]:
            self.skip_animations = True
        if (
            config["from_animation_number"]
            and self.num_plays < config["from_animation_number"]
        ):
            self.skip_animations = True
        if (
            config["upto_animation_number"]
            and self.num_plays > config["upto_animation_number"]
        ):
            self.skip_animations = True
            raise EndSceneEarlyException()

    def scene_finished(self, scene):
        # If no animations in scene, render an image instead
        if self.num_plays:
            self.file_writer.finish()
        elif config.write_to_movie:
            config.save_last_frame = True
            config.write_to_movie = False
        else:
            self.static_image = None
            self.update_frame(scene)

        if config["save_last_frame"]:
            self.static_image = None
            self.update_frame(scene)
            self.file_writer.save_final_image(self.camera.get_image())
