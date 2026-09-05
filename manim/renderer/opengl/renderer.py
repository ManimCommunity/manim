from __future__ import annotations

import contextlib
import itertools as it
import threading
import time
import typing
from typing import TYPE_CHECKING, Any

import moderngl
import numpy as np
from moderngl import Framebuffer
from PIL import Image

from manim import config
from manim.mobject.opengl.opengl_mobject import (
    OpenGLMobject,
)
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject
from manim.typing import Point3D
from manim.utils.caching import handle_caching_play
from manim.utils.color import color_to_rgba
from manim.utils.exceptions import EndSceneEarlyException

from ...constants import *
from ...scene.scene_file_writer import SceneFileWriter
from ..protocol import RendererCapabilities
from .shader import Mesh, Shader
from .vectorized_mobject_rendering import (
    render_opengl_vectorized_mobject_fill,
    render_opengl_vectorized_mobject_stroke,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from manim._config.render_session import RenderSessionSpec
    from manim.animation.animation import Animation
    from manim.mobject.mobject import Mobject, _AnimationBuilder
    from manim.scene.scene import Scene
    from manim.scene.scene_file_writer import _SceneFileWriterSettings
    from manim.typing import (
        FloatRGBA,
        RGBAPixelArray,
    )
    from manim.utils.color.core import ParsableManimColor

    from .window import Window

from .camera import OpenGLCamera

__all__ = ["OpenGLRenderer"]


class OpenGLRenderer:
    """
    An OpenGL-based renderer.

    Attributes
    ----------
    animation_elapsed_time : float
        The elapsed time of the current animation.
    animation_start_time : float
        The start time of the current animation.
    animations_hashes : list[str | None]
        List of animation hashes for caching.
    anti_alias_width : float
        The width used for anti-aliasing in pixel units.
    background_color : FloatRGBA
        The background color of the renderer.
    camera : OpenGLCamera
        The camera used for rendering.
    num_plays : float
        The number of animation plays executed.
    path_to_texture_id : dict[str, int]
        Mapping from texture file paths to OpenGL texture IDs.
    pressed_keys : set[int]
        Set of currently pressed key codes.
    skip_animations : bool
        Whether animations are currently being skipped.
    time : float
        The total elapsed time for the renderer.
    window : Window | None
        The window used for previewing, if any.
    """

    capabilities = RendererCapabilities(live_preview=True)

    def __init__(
        self,
        file_writer_class: type[SceneFileWriter] = SceneFileWriter,
        skip_animations: bool = False,
    ) -> None:
        """Initializes the OpenGLRenderer.

        Parameters
        ----------
        file_writer_class : type[SceneFileWriter], optional
            The class to use for writing scene files, by default SceneFileWriter.
        skip_animations : bool, optional
            Whether to skip animations during rendering, by default False.
        """
        # Measured in pixel widths, used for vector graphics
        self.anti_alias_width = 1.5
        self._file_writer_class = file_writer_class

        self._original_skipping_status = skip_animations
        self.skip_animations = skip_animations
        self.animation_start_time = 0.0
        self.animation_elapsed_time = 0.0
        self.time = 0.0
        self.animations_hashes: list[str | None] = []
        self.num_plays = 0

        self.camera = OpenGLCamera()
        self.pressed_keys: set[int] = set()
        self.window: Window | None = None
        self.path_to_texture_id: dict[str, int] = {}
        self.background_color = config["background_color"]

    def init_scene(
        self,
        scene: Scene,
        session_spec: RenderSessionSpec,
        file_writer_settings: _SceneFileWriterSettings,
    ) -> None:
        """
        Initializes the OpenGL rendering context and related resources
        for the given scene.

        Set up:
        - the file writer
        - the background color
        - the OpenGL context
        - the window (if needed)

        Parameters
        ----------
        scene : Scene
            The scene to be rendered
        """
        self.partial_movie_files: list[str | None] = []
        self.file_writer: SceneFileWriter = self._file_writer_class(
            file_writer_settings,
        )
        self.scene = scene

        self.background_color = config["background_color"]
        if self.should_create_window(session_spec):
            from .window import Window

            self.window = Window(self)
            self.context = self.window.ctx
            self.frame_buffer_object = self.context.detect_framebuffer()
        else:
            # self.window = None
            try:
                self.context = moderngl.create_context(standalone=True)
            except Exception:
                self.context = moderngl.create_context(
                    standalone=True,
                    backend="egl",
                )
            self.frame_buffer_object = self.get_frame_buffer_object(self.context, 0)
            self.frame_buffer_object.use()
        self._context_thread = threading.get_ident()
        self._capturing_image = False
        self.context.enable(moderngl.BLEND)
        self.context.wireframe = config["enable_wireframe"]
        self.context.blend_func = (
            moderngl.SRC_ALPHA,
            moderngl.ONE_MINUS_SRC_ALPHA,
            moderngl.ONE,
            moderngl.ONE,
        )

    def should_create_window(self, session_spec: RenderSessionSpec) -> bool:
        """
        Determine whether a window should be created for rendering
        based on the current configuration.

        """
        return session_spec.presentation.live_preview

    def get_pixel_shape(self) -> tuple[int, int] | None:
        """
        Retrieve the pixel dimensions of the current frame buffer object (2D).

        Returns
        -------
        width : int
            The width of the frame buffer in pixels.
        height : int
            The height of the frame buffer in pixels.
        """
        frame_buffer: Framebuffer | None = getattr(self, "frame_buffer_object", None)
        if frame_buffer is None:
            return None
        _, _, pixel_width, pixel_height = frame_buffer.viewport
        return pixel_width, pixel_height

    def refresh_perspective_uniforms(self, camera: OpenGLCamera) -> None:
        """
        Update the perspective-related uniform variables used in the
        OpenGL renderer based on the current camera settings.

        Parameters
        ----------
        camera : OpenGLCamera
            The camera object from which to extract perspective and lighting information.

        Raises
        ------
        ValueError
            If the renderer's pixel shape is not available.
        """
        pixel_shape = self.get_pixel_shape()
        if pixel_shape is None:
            msg = "Pixel shape is None, cannot refresh perspective uniforms."
            raise ValueError(msg)

        pixel_width, pixel_height = pixel_shape
        frame_width, frame_height = camera.get_shape()
        # TODO, this should probably be a mobject uniform, with
        # the camera taking care of the conversion factor
        anti_alias_width = self.anti_alias_width / (pixel_height / frame_height)
        # Orient light
        rotation = camera.inverse_rotation_matrix
        light_pos: Point3D = camera.light_source.get_location()
        light_pos = np.dot(rotation, light_pos)

        self.perspective_uniforms = {
            "frame_shape": camera.get_shape(),
            "anti_alias_width": anti_alias_width,
            "camera_center": tuple(camera.get_center()),
            "camera_rotation": tuple(np.array(rotation).T.flatten()),
            "light_source_position": tuple(light_pos),
            "focal_distance": camera.get_focal_distance(),
        }

    def render_mobject(self, mobject: OpenGLMobject | OpenGLVMobject) -> None:
        """
        Render an OpenGL mobject (either OpenGLMobject or OpenGLVMobject)
        using the appropriate shaders and rendering pipeline.

        Parameters
        ----------
        mobject : OpenGLMobject | OpenGLVMobject
            The mobject to render. Must be an instance of OpenGLMobject or OpenGLVMobject.

        Raises
        ------
        TypeError
            If a shader texture is not a moderngl.Uniform or moderngl.UniformBlock.
        """
        if isinstance(mobject, OpenGLVMobject):
            if config["use_projection_fill_shaders"]:
                render_opengl_vectorized_mobject_fill(self, mobject)

            if config["use_projection_stroke_shaders"]:
                render_opengl_vectorized_mobject_stroke(self, mobject)

        shader_wrapper_list = mobject.get_shader_wrapper_list()
        # Convert ShaderWrappers to Meshes.
        for shader_wrapper in shader_wrapper_list:
            folder = shader_wrapper.shader_folder
            shader = Shader(
                context=self.context, name=str(folder) if folder is not None else None
            )

            # Set textures.
            for name, path in shader_wrapper.texture_paths.items():
                tid = self.get_texture_id(str(path))
                shader_texture = shader.shader_program[name]
                if not isinstance(
                    shader_texture, (moderngl.Uniform, moderngl.UniformBlock)
                ):
                    msg = (
                        f"Shader texture must be a uniform, got {type(shader_texture)}"
                    )
                    raise TypeError(msg)
                shader_texture.value = tid

            # Set uniforms.
            for name, value in it.chain(
                shader_wrapper.uniforms.items(),
                self.perspective_uniforms.items(),
            ):
                with contextlib.suppress(KeyError):
                    shader.set_uniform(name, value)
            try:
                # TODO: make the type of 'camera' generic in the 'Scene' class
                # to avoid the cast here
                cam = typing.cast("OpenGLCamera", self.scene.camera)
                shader.set_uniform("u_view_matrix", cam.formatted_view_matrix)
                shader.set_uniform("u_projection_matrix", cam.projection_matrix)
            except KeyError:
                pass

            # Set depth test.
            if shader_wrapper.depth_test:
                self.context.enable(moderngl.DEPTH_TEST)
            else:
                self.context.disable(moderngl.DEPTH_TEST)

            # Render.
            vert_indices = shader_wrapper.vert_indices
            mesh = Mesh(
                shader,
                shader_wrapper.vert_data,
                indices=np.asarray(vert_indices) if vert_indices is not None else None,
                use_depth_test=shader_wrapper.depth_test,
                primitive=mobject.render_primitive,
            )
            mesh.set_uniforms(self)
            mesh.render()

    def get_texture_id(self, path: str) -> int:
        """
        Retrieves the OpenGL texture ID associated with the given image file path.

        Automatically creates a new texture it it has not been loaded before.

        Parameters
        ----------
        path : str
            The file path to the texture image.

        Returns
        -------
        int
            The OpenGL texture ID corresponding to the given path.
        """
        return (
            self.path_to_texture_id[path]
            if path in self.path_to_texture_id
            else self._create_texture(path)
        )

    def _create_texture(self, image_path: str) -> int:
        """
        Create an OpenGL texture from the given image file path, get its texture ID,
        and store it in `self.path_to_texture_id[image_path]`.

        Parameters
        ----------
        image_path : str
            The file path to the image to be loaded as a texture.

        Returns
        -------
        int
            The texture ID assigned to the newly created texture.
        """
        with Image.open(image_path) as img:
            tid = len(self.path_to_texture_id)

            # grayscale image
            if img.mode == "L":
                components = 1
                swizzle = "RRR1"
            else:
                # convert everything to RGBA for consistency
                img = img.convert("RGBA")
                components = 4
                swizzle = "RGBA"

            texture = self.context.texture(
                size=img.size,
                components=components,
                data=img.tobytes(),
            )
        texture.repeat_x = False
        texture.repeat_y = False
        texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        texture.swizzle = swizzle
        texture.use(location=tid)
        self.path_to_texture_id[image_path] = tid
        return tid

    def update_skipping_status(self) -> None:
        """
        Check and update the skipping status for the current animation
        (self.skip_animations flag) based on the configuration settings.

        Parameters
        ----------
        None

        Raises
        ------
        EndSceneEarlyException
            If the number of played animations exceeds the configured upper bound.
        """
        # there is always at least one section -> no out of bounds here
        if self.file_writer.sections[-1].skip_animations:
            self.skip_animations = True
        if self.file_writer.output_spec.is_still:
            self.skip_animations = True
        if (
            config.from_animation_number > 0
            and self.num_plays < config.from_animation_number
        ):
            self.skip_animations = True
        if (
            config.upto_animation_number >= 0
            and self.num_plays > config.upto_animation_number
        ):
            self.skip_animations = True
            raise EndSceneEarlyException()

    @handle_caching_play
    def play(
        self,
        scene: Scene,
        *animations: Animation | Mobject | _AnimationBuilder,
        **kwargs: Any,
    ) -> None:
        """
        Plays the given animations or mobjects in the specified scene.

        "Playing" here refers to the process of compiling animation data,
        beginning the animations, updating frames, and finalizing the animation
        in the context of the renderer.

        Parameters
        ----------
        scene Scene
            The scene in which to play the animations.
        *animations Animation | Mobject | _AnimationBuilder
            The animations, mobjects, or animation builders to play.
        **kwargs Any
            Additional keyword arguments to pass to the animation compilation.
        """
        # TODO: Handle data locking / unlocking.
        self.animation_start_time = time.time()
        self.file_writer.begin_animation(
            not self.skip_animations,
            animation_index=self.num_plays,
        )

        scene.compile_animation_data(*animations, **kwargs)
        scene.begin_animations()
        if scene.is_current_animation_frozen_frame():
            self.update_frame(scene)

            output = self.file_writer.output_spec
            if not self.skip_animations and (
                output.is_video or output.is_image_sequence
            ):
                self.file_writer.write_frame(
                    self.get_frame(),
                    repeat=int(config.frame_rate * scene.duration),
                )

            if self.window is not None:
                self.window.swap_buffers()
                while time.time() - self.animation_start_time < scene.duration:
                    pass
            self.animation_elapsed_time = scene.duration

        else:
            scene.play_internal()

        self.file_writer.end_animation(not self.skip_animations)
        self.time += scene.duration
        self.num_plays += 1

    def clear_screen(self) -> None:
        """
        Clears the current frame buffer and updates the display window
        accordingly.

        The screen is cleared using the background color specified
        in the renderer.
        """
        self.frame_buffer_object.clear(*self.background_color)
        if self.window is None:
            return
        self.window.swap_buffers()

    def render(
        self, scene: Scene, frame_offset: float, moving_mobjects: list[Mobject]
    ) -> None:
        """
        Renders a single frame of the given scene using OpenGL.

        Parameters
        ----------
        scene : Scene
            The scene to render.
        frame_offset : float
            The time offset for the current frame in seconds. If no window is present,
            this parameter is ignored, and a frame is a true snapshot of
            the scene at the current time.
        moving_mobjects : list[Mobject]
            List of mobjects that are currently moving and need to be updated.
            Not used at all, kept for compatibility with other renderers.

        Notes
        -----
        - Updates the frame for the scene.
        - If animations are skipped, the method returns early.
        - Writes the current frame using the file writer.
        - If a window is present, swaps buffers and continues
          updating frames until the animation elapsed time reaches the frame offset.
        """
        self.update_frame(scene)

        if self.skip_animations:
            return

        output = self.file_writer.output_spec
        if output.is_video or output.is_image_sequence:
            self.file_writer.write_frame(self.get_frame())

        if self.window is not None:
            self.window.swap_buffers()
            while self.animation_elapsed_time < frame_offset:
                self.update_frame(scene)
                self.window.swap_buffers()

    def update_frame(self, scene: Scene) -> None:
        """
        Update and render the current frame for the given scene.

        Performs the following steps:
        1. Clear the frame buffer with the background color.
        2. Refresh camera perspective uniforms for rendering.
        3. Iterate through all mobjects in the scene, rendering those marked for display.
        4. Iterate through all mesh objects in the scene, setting their uniforms and rendering them.
        5. Update the elapsed animation time.

        Parameters
        ----------
        scene : Scene
            The scene to render the frame for.
        """
        self._draw_scene(scene)
        self.animation_elapsed_time = time.time() - self.animation_start_time

    def _draw_scene(self, scene: Scene) -> None:
        """Draw current objects and meshes without updating animation timing."""
        self.frame_buffer_object.clear(*self.background_color)

        # TODO: make the type of 'camera' generic in the 'Scene' class
        # to avoid the cast here
        cam = typing.cast("OpenGLCamera", scene.camera)
        self.refresh_perspective_uniforms(cam)

        for mobject in scene.mobjects:
            if not mobject.should_render:
                continue

            # TODO: make the type of 'mobject' generic in the 'Scene' class
            # to avoid the cast here
            mobj = typing.cast("OpenGLMobject | OpenGLVMobject", mobject)
            self.render_mobject(mobj)

        for obj in scene.meshes:
            for mesh in obj.get_meshes():
                mesh.set_uniforms(self)
                mesh.render()

    def _get_scene_image(self, scene: Scene) -> Image.Image:
        """Capture on the context's owning thread without touching its live target."""
        if threading.get_ident() != self._context_thread:
            raise RuntimeError(
                "OpenGL scene images must be requested on the render thread."
            )
        if self._capturing_image:
            raise RuntimeError("Recursive OpenGL scene image capture is not supported.")
        target = self.frame_buffer_object
        bound = self.context.fbo
        viewport = self.context.viewport
        size = target.size
        self._capturing_image = True
        try:
            with contextlib.ExitStack() as resources:
                color = self.context.texture(size, components=4)
                resources.callback(color.release)
                depth = self.context.depth_renderbuffer(size)
                resources.callback(depth.release)
                frame = self.context.framebuffer(color, depth)
                resources.callback(frame.release)
                try:
                    self.frame_buffer_object = frame
                    frame.use()
                    self._draw_scene(scene)
                    return Image.fromarray(self.get_frame())
                finally:
                    self.frame_buffer_object = target
                    (target if bound is None else bound).use()
                    self.context.viewport = viewport
        finally:
            self._capturing_image = False

    def scene_finished(self, scene: Scene) -> None:
        """Finalize configured output for the scene.

        Parameters
        ----------
        scene
            The scene that has finished rendering.
        """
        output = self.file_writer.output_spec
        if self.num_plays > 0 and (output.is_video or output.is_image_sequence):
            self.file_writer.finish()
        elif self.num_plays == 0:
            # Keep the framebuffer useful for direct renderer access and
            # graphical tests even when no media artifact was requested.
            self.update_frame(scene)

        if self.should_save_last_frame():
            if self.num_plays > 0:
                self.update_frame(scene)
            self.file_writer.save_image(self.get_frame())

    def should_save_last_frame(self) -> bool:
        """
        Determine whether the last frame of the scene should be saved.

        This is true for explicit last-frame PNG output and for automatic video
        output when the scene has no play calls. Interactive scenes do not use
        the automatic fallback.
        """
        output = self.file_writer.output_spec
        if output.is_still:
            return True
        if self.scene.interactive_mode:
            return False
        return self.num_plays == 0 and output.fallback_to_still

    def get_image(self) -> Image.Image:
        """
        Get the current OpenGL frame buffer as a PIL Image.

        Returns
        -------
        Image.Image
            The image representation of the current frame buffer.

        Raises
        ------
        ValueError
            If the pixel shape cannot be determined.

        Notes
        -----
        The image is constructed from raw RGBA buffer data, with the
        origin at the bottom-left.
        """
        raw_buffer_data = self.get_raw_frame_buffer_object_data()
        pixel_shape = self.get_pixel_shape()
        if pixel_shape is None:
            msg = "Pixel shape is None, cannot get image."
            raise ValueError(msg)

        image = Image.frombytes(
            "RGBA",  # mode (rgb, a for alpha (transparency)))
            pixel_shape,  # size
            raw_buffer_data,  # data
            "raw",  # decoder_name
            # *args for the decoder
            "RGBA",  # raw mode
            0,  # stride (O = no extra padding)
            -1,  # orientation (-1 = bottom to top, 1 = top to bottom)
        )
        return image

    def save_static_frame_data(
        self, scene: Scene, static_mobjects: Iterable[Mobject]
    ) -> None:
        pass

    def get_frame_buffer_object(
        self, context: moderngl.Context, samples: int = 0
    ) -> Framebuffer:
        """
        Creates and returns a framebuffer object configured with color
        and depth attachments.

        Parameters
        ----------
        context : moderngl.Context
            The ModernGL context used to create the framebuffer and
            its attachments.
        samples : int, optional
            The number of samples for multisample anti-aliasing (MSAA)[1]_.
            Default is 0 (no MSAA).

        Returns
        -------
        Framebuffer
            A framebuffer object with a color texture attachment and
            a depth renderbuffer attachment, both sized according to
            the current configuration's pixel width and height.

        Notes
        -----
        Framebuffer's color attachment is supposed RGBA.
        Pixel dimensions are taken from the global config of Manim.

        References
        ----------
        .. [1] Wikipedia, "Multisample anti-aliasing",
               https://en.wikipedia.org/wiki/Multisample_anti-aliasing
        """
        pixel_width = config["pixel_width"]
        pixel_height = config["pixel_height"]
        num_channels = 4
        return context.framebuffer(
            color_attachments=context.texture(
                (pixel_width, pixel_height),
                components=num_channels,
                samples=samples,
            ),
            depth_attachment=context.depth_renderbuffer(
                (pixel_width, pixel_height),
                samples=samples,
            ),
        )

    def get_raw_frame_buffer_object_data(self, dtype: str = "f1") -> bytes:
        """
        Get the raw data from the current frame buffer object as bytes.

        This method reads the pixel data from the frame buffer object using the specified data type.
        The data is read with 4 color channels (typically RGBA).

        Args:
            dtype (str, optional): The data type to use when reading the buffer.
            Defaults to "f1" (i.e., float with 1 byte).

        Returns:
            bytes: The raw pixel data from the frame buffer object.
        """
        # Copy blocks from the fbo_msaa to the drawn fbo using Blit
        # pw, ph = self.get_pixel_shape()
        # gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.fbo_msaa.glo)
        # gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self.fbo.glo)
        # gl.glBlitFramebuffer(
        #     0, 0, pw, ph, 0, 0, pw, ph, gl.GL_COLOR_BUFFER_BIT, gl.GL_LINEAR
        # )
        num_channels = 4
        ret: bytes = self.frame_buffer_object.read(
            viewport=self.frame_buffer_object.viewport,
            components=num_channels,
            dtype=dtype,
        )
        return ret

    def get_frame(self) -> RGBAPixelArray:
        """
        Get the current frame buffer as a Numpy array of RGBA pixel values.

        Returns
        -------
        RGBAPixelArray
            A Numpy array of shape (height, width, 4) containing the
            RGBA pixel data of the current frame, with dtype uint8.

        Raises
        ------
        ValueError
            If the pixel shape cannot be determined.
        """
        # get current pixel values as numpy data in order to test output
        raw = self.get_raw_frame_buffer_object_data(dtype="f1")
        pixel_shape = self.get_pixel_shape()
        if pixel_shape is None:
            msg = "Pixel shape is None, cannot get frame."
            raise ValueError(msg)

        result_dimensions = (pixel_shape[1], pixel_shape[0], 4)
        np_buf = np.frombuffer(raw, dtype="uint8").reshape(result_dimensions)
        return np.flipud(np_buf).copy()

    # Returns offset from the bottom left corner in pixels.
    # top_left flag should be set to True when using a GUI framework
    # where the (0,0) is at the top left: e.g. PySide6
    def pixel_coords_to_space_coords(
        self, px: float, py: float, relative: bool = False, top_left: bool = False
    ) -> Point3D:
        """
        Converts pixel coordinates to space (scene) coordinates.

        top_left flag should be set to True when using a GUI framework
        where the (0,0) is at the top left: e.g. PySide6.

        Parameters
        ----------
        px : float
            The x-coordinate in pixel space.
        py : float
            The y-coordinate in pixel space.
        relative : bool, optional
            If True, returns coordinates relative to the frame (normalized to [-1, 1]).
            If False, returns absolute space coordinates. Default is False.
        top_left : bool, optional
            If True, treats the origin (0, 0) as the top-left corner of the pixel space.
            If False, treats the origin as the bottom-left. Default is False.

        Returns
        -------
        Point3D
            The corresponding coordinates in space as a NumPy array of shape (3,).

        Notes
        -----
        If the pixel shape is not available, returns the origin [0, 0, 0].
        """
        pixel_shape = self.get_pixel_shape()
        if pixel_shape is None:
            return typing.cast(Point3D, np.array([0.0, 0.0, 0.0]))
        pixel_width, pixel_height = pixel_shape
        frame_height = config["frame_height"]
        frame_center = self.camera.get_center()
        if relative:
            # relative -> just normalize to [-1, 1]
            return 2 * np.array([px / pixel_width, py / pixel_height, 0])

        scale = frame_height / pixel_height
        y_direction = -1 if top_left else 1

        return typing.cast(
            Point3D,
            frame_center
            + scale
            * np.array(
                [(px - pixel_width / 2), y_direction * (py - pixel_height / 2), 0.0]
            ),
        )

    @property
    def background_color(self) -> FloatRGBA:
        """The background color of the renderer (RGBA format)."""
        return self._background_color

    @background_color.setter
    def background_color(self, value: ParsableManimColor) -> None:
        self._background_color = color_to_rgba(value, 1.0)
