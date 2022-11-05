from __future__ import annotations

import numpy as np
from tqdm import tqdm as ProgressDisplay

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable, Iterable

    from PIL.Image import Image


class Scene:
    r"""A Scene is the canvas used to draw and animate objects.


    """

    def __init__(
        self,
        *,
        camera_class: Type[Camera] | None = None,
        always_update_mobjects: bool = False,
        random_seed: int | None = 0,
        render_from_animation_number: int | None = None,
        render_to_animation_number: int | None = None,
        live_preview: bool = True,
        preview_window_config: dict | None = None,
        show_render_progress: bool = True,
        max_num_saved_states: int = 50,
    ):
        if live_preview:
            from manim.window import Window  # fix import
            preview_window_config = preview_window_config or {}
            self.window = Window(scene=self, **preview_window_config)
        else:
            self.window = None

        camera_class = OpenGLCamera if config.renderer == "opengl" else CairoCamera
        self.camera: Type[Camera] = camera_class()
        self.file_writer: SceneFileWriter = SceneFileWriter(self)
        self.mobjects: list[Mobject] = []
        self.num_plays: int = 0
        self.time: float = 0
        self.skip_rendering: bool = config.skipping_status  # TODO: not a thing yet, config replaces original_skipping_status?

        self.render_from_animation_number = render_from_animation_number
        self.render_to_animation_number = render_to_animation_number
        if render_from_animation_number is not None:
            self.skip_rendering = True

        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def __str__(self) -> str:
        return self.__class__.__name__

    def render(self) -> None:
        """Render this scene.

        This is the entry point for rendering scenes. It calls
        the three fundamental life cycle hooks of a scene,
        the methods :meth:`.setup`, :meth:`.construct`, and
        :meth:`.tear_down`, and takes care of the required
        communication with the file writer object.
        """
        self.file_writer.begin()

        self.setup()
        try:
            self.construct()
        except EndScene:
            pass
        except KeyboardInterrupt:
            # in case the scene was used interactively, end gracefully
            # upon Ctrl-C.
            print("", end="\r")
            self.file_writer.ended_with_interrupt = True

        self.tear_down()

    def setup(self) -> None:
        """First method run within the render life cycle of a scene.

        This method is intended for preparing custom attributes
        of the scene that are used during :meth:`.construct`.
        """
        pass

    def construct(self) -> None:
        """The central entry point for writing animations.

        This is the second method run within the render life cycle
        of a scene. It is intended to hold your animation script.
        """
        pass

    def tear_down(self) -> None:
        """Ends the render life cycle of a scene.

        Communicates the end of the render life cycle to the
        file writer, and takes care of closing the live preview
        window.
        """
        self.stop_skipping()
        self.file_writer.finish()
        if self.window:
            self.window.destroy()
            self.window = None

    # methods interacting with the camera
    def get_image(self) -> Image:
        """Get the current image from the camera."""
        return self.camera.get_image()

    def show(self) -> None:
        """Render the current frame and display it in the default
        system image viewer.
        """
        self.update_frame(ignore_skipping=True)
        self.get_image().show()

    def update_frame(
        self,
        dt: float = 0,
        ignore_skipping: bool = False,
    ) -> None:
        """Advance the scene time by the specified amount and
        capture a new frame.
        """
        self.increment_time(dt)
        self.update_mobjects(dt)
        if self.skip_rendering and not ignore_skipping:
            return

        if self.is_window_closing():
            raise EndScene()

        if self.window:
            self.window.clear()
        self.camera.clear()
        self.camera.capture(*self.mobjects)

        if self.window:
            self.window.swap_buffers()
            real_time = time.time() - self.real_start_time
            if real_time < self.time:
                self.update_frame(dt=0)

    def emit_frame(self) -> None:
        """Pass the latest captured frame to the file writer."""
        if not self.skip_rendering:
            self.file_writer.write_frame(self.camera)







