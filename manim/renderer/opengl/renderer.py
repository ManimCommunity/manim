from __future__ import annotations

import contextlib
import itertools as it
import threading
import time
import typing
import weakref
from typing import TYPE_CHECKING, Any

import moderngl
import numpy as np
from moderngl import Framebuffer
from PIL import Image

from manim import config, logger
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
from .shader import Mesh, Shader, shader_program_cache
from .vectorized_mobject_rendering import (
    render_opengl_vectorized_mobject_fill,
    render_opengl_vectorized_mobject_stroke,
)
from .window_settings import _WindowSettings

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from manim._config.render_session import RenderSessionSpec
    from manim.animation.animation import Animation
    from manim.mobject.mobject import Mobject, _AnimationBuilder
    from manim.scene.scene import Scene
    from manim.typing import (
        FloatRGBA,
        RGBAPixelArray,
    )
    from manim.utils.color.core import ParsableManimColor

    from .window import Window

from .camera import OpenGLCamera

__all__ = ["OpenGLRenderer"]

# Remember the active Manim renderer on each thread so captures can restore it.
_active_context = threading.local()


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
        self._context: moderngl.Context | None = None
        self._frame_buffer_object: Framebuffer | None = None
        self._context_thread: int | None = None
        self._capturing_image = False
        self._closed = False
        self._retiring = False
        self._resources = contextlib.ExitStack()
        self._textures: list[moderngl.Texture] = []
        self._previous_renderer: weakref.ReferenceType[OpenGLRenderer] | None = None

    def init_scene(
        self,
        scene: Scene,
        session_spec: RenderSessionSpec,
    ) -> None:
        """
        Attach a scene and save its pixel dimensions and preview-window settings.

        :meth:`open` uses these settings to create the OpenGL context and
        framebuffer. The manager calls it before the scene's ``setup()`` method;
        an explicit request for GPU resources, such as :attr:`context`, opens
        them sooner.

        Parameters
        ----------
        scene : Scene
            The scene to be rendered
        """
        self._ensure_not_closed()
        if hasattr(self, "scene"):
            raise RuntimeError("This renderer is already bound to a Scene.")
        self.partial_movie_files: list[str | None] = []
        self.scene = scene

        self.background_color = config["background_color"]
        self._pixel_size = (int(config.pixel_width), int(config.pixel_height))
        self._wireframe = config.enable_wireframe
        self._window_settings = (
            _WindowSettings.from_config(config)
            if self.should_create_window(session_spec)
            else None
        )

    def _ensure_not_closed(self) -> None:
        if self._closed or self._retiring:
            raise RuntimeError("The OpenGL renderer is closed or retiring.")

    def open(self) -> None:
        """Create or activate the scene's OpenGL context.

        The first call creates the context, framebuffer, and preview window if
        requested. Further calls activate the same context. Use and close these
        resources on the thread that first opened them.
        """
        self._ensure_not_closed()
        if self._context is not None:
            if isinstance(self._context.mglo, moderngl.InvalidObject):
                raise RuntimeError("The OpenGL context was released externally.")
            if self.window is not None and self.window._window.context is None:
                raise RuntimeError("The OpenGL window was destroyed externally.")
            if self._context_thread != threading.get_ident():
                raise RuntimeError(
                    "OpenGL resources must be used on their owning thread."
                )
            self._activate_context()
            return
        if not hasattr(self, "scene"):
            raise RuntimeError("Bind an OpenGL scene before opening its resources.")
        previous = getattr(_active_context, "renderer", None)
        resources = contextlib.ExitStack()
        try:
            window = None
            if self._window_settings is not None:
                from .window import Window

                window = Window(self, _settings=self._window_settings)
                resources.callback(window.close)
                # Closing the window also releases its context and default framebuffer.
                context = window.ctx
                frame = context.detect_framebuffer()
            else:
                try:
                    context = moderngl.create_context(standalone=True)
                except Exception:
                    context = moderngl.create_context(standalone=True, backend="egl")
                resources.callback(context.release)
                frame = self.get_frame_buffer_object(context, 0, size=self._pixel_size)
                for color in frame.color_attachments:
                    resources.callback(color.release)
                if frame.depth_attachment is not None:
                    resources.callback(frame.depth_attachment.release)
                resources.callback(frame.release)
                frame.use()
            context.enable(moderngl.BLEND)
            context.wireframe = self._wireframe
            context.blend_func = (
                moderngl.SRC_ALPHA,
                moderngl.ONE_MINUS_SRC_ALPHA,
                moderngl.ONE,
                moderngl.ONE,
            )
        except BaseException:
            try:
                resources.close()
            except BaseException:
                logger.exception("Failed to roll back OpenGL initialization")
            try:
                self._restore_renderer(previous)
            except BaseException:
                logger.exception("Failed to restore the previous OpenGL context")
            raise
        # Store the resources after initialization succeeds.
        self.window = window
        self._context = context
        self._frame_buffer_object = frame
        self._context_thread = threading.get_ident()
        self._capturing_image = False
        self._resources = resources.pop_all()
        self._previous_renderer = previous
        _active_context.renderer = weakref.ref(self)

    def _activate_context(self) -> None:
        if self.window is not None:
            self.window._window.switch_to()
        else:
            assert self._context is not None
            self._context.__enter__()
        _active_context.renderer = weakref.ref(self)

    @property
    def context(self) -> moderngl.Context:
        self.open()
        assert self._context is not None
        return self._context

    @property
    def frame_buffer_object(self) -> Framebuffer:
        self.open()
        assert self._frame_buffer_object is not None
        return self._frame_buffer_object

    @frame_buffer_object.setter
    def frame_buffer_object(self, frame: Framebuffer) -> None:
        self._ensure_not_closed()
        self._frame_buffer_object = frame

    def close(self) -> None:
        """Release GPU resources and close the preview window.

        Use :meth:`.Manager.close` for scene cleanup: it stops encoding jobs
        before closing the renderer. To render another scene, create a new
        renderer. :meth:`.Scene.get_image` can still draw ordinary mobjects with
        a temporary context; GPU-backed meshes require their original context
        to remain open.
        """
        if self._closed:
            return
        if self._capturing_image:
            raise RuntimeError("Cannot close OpenGL resources during image capture.")
        if self._context is not None and self._context_thread != threading.get_ident():
            raise RuntimeError(
                "OpenGL resources must be closed on their owning thread."
            )
        if self._context is None:
            self._closed = True
            return
        current = getattr(_active_context, "renderer", None)
        previous = (
            self._previous_renderer
            if current is not None and current() is self
            else current
        )
        failures: list[BaseException] = []

        def release(callback: Callable[[], Any]) -> None:
            try:
                callback()
            except BaseException as error:
                failures.append(error)

        # Scene.interact may already have closed the window and released its GPU
        # objects. Deleting them again could affect a different active context.
        host_alive = (
            self._context is not None
            and not isinstance(self._context.mglo, moderngl.InvalidObject)
            and (self.window is None or self.window._window.context is not None)
        )
        if self._context is not None and host_alive:
            self._activate_context()
        self._retiring = True
        for name, program in list(shader_program_cache.items()):
            if program.ctx is self._context:
                del shader_program_cache[name]
                if host_alive:
                    release(program.release)
        if host_alive:
            for texture in reversed(self._textures):
                release(texture.release)
        self._textures.clear()
        if host_alive:
            release(self._resources.close)
        else:
            self._resources.pop_all()
            if self.window is not None and self.window._window.context is not None:
                release(self.window.close)
        self.path_to_texture_id.clear()
        self.pressed_keys.clear()
        self._resources = contextlib.ExitStack()
        host_still_alive = (
            not isinstance(self._context.mglo, moderngl.InvalidObject)
            if self.window is None
            else self.window._window.context is not None
        )
        if failures and host_still_alive:
            # Keep the context reachable so close() can retry. Closing it releases
            # any remaining GPU objects; rendering stays disabled until then.
            self._resources.callback(
                self._context.release if self.window is None else self.window.close
            )
        else:
            self._context = None
            self._frame_buffer_object = None
            self._context_thread = None
            self.window = None
            self._retiring = False
            self._closed = True
        _active_context.renderer = None
        release(lambda: self._restore_renderer(previous))
        if failures:
            for secondary in failures[1:]:
                logger.error("Additional OpenGL cleanup failure", exc_info=secondary)
            raise failures[0]

    @staticmethod
    def _restore_renderer(
        reference: weakref.ReferenceType[OpenGLRenderer] | None,
    ) -> None:
        prior = reference() if reference is not None else None
        if (
            prior is not None
            and not prior._closed
            and not prior._retiring
            and prior._context is not None
            and not isinstance(prior._context.mglo, moderngl.InvalidObject)
            and (prior.window is None or prior.window._window.context is not None)
        ):
            prior.open()

    @property
    def file_writer(self) -> SceneFileWriter:
        """Return the scene manager's file writer, creating it if needed."""
        return self.scene._get_manager().file_writer

    def should_create_window(self, session_spec: RenderSessionSpec) -> bool:
        """Return whether the scene's saved settings request a live-preview window."""
        return session_spec.presentation.live_preview

    def get_pixel_shape(self) -> tuple[int, int] | None:
        """
        Return the pixel dimensions of the current framebuffer.

        Returns
        -------
        tuple[int, int] | None
            ``(width, height)`` in pixels, or ``None`` when no framebuffer is open.
        """
        frame_buffer = self._frame_buffer_object
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
        self.open()
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
        self.open()
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
        self._ensure_not_closed()
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
        self._textures.append(texture)
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
        self.open()
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
        """Draw a snapshot separately from the live-preview framebuffer.

        An existing context is used on its rendering thread. Otherwise, ordinary
        mobjects are drawn with a temporary context and renderer.
        """
        if (
            self._context_thread is not None
            and threading.get_ident() != self._context_thread
        ):
            raise RuntimeError(
                "OpenGL scene images must be requested on the render thread."
            )
        if self._capturing_image:
            raise RuntimeError("Recursive OpenGL scene image capture is not supported.")
        if self._context is None:
            if scene.meshes:
                raise RuntimeError(
                    "GPU-backed meshes require a live OpenGL context for inspection."
                )
            temporary = OpenGLRenderer()
            temporary.scene = scene
            temporary.camera = self.camera
            temporary._background_color = self._background_color.copy()
            temporary._pixel_size = self._pixel_size
            temporary._wireframe = self._wireframe
            temporary._window_settings = None
            self._capturing_image = True
            try:
                try:
                    temporary._draw_scene(scene)
                    image = temporary.get_image()
                except BaseException:
                    try:
                        temporary.close()
                    except BaseException:
                        logger.exception(
                            "Failed to close the temporary OpenGL image scope"
                        )
                    raise
                temporary.close()
                return image
            finally:
                self._capturing_image = False
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
        self._ensure_not_closed()

    def get_frame_buffer_object(
        self,
        context: moderngl.Context,
        samples: int = 0,
        *,
        size: tuple[int, int] | None = None,
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
        pixel_width, pixel_height = (
            (config["pixel_width"], config["pixel_height"]) if size is None else size
        )
        num_channels = 4
        resources = contextlib.ExitStack()
        try:
            color = context.texture(
                (pixel_width, pixel_height), components=num_channels, samples=samples
            )
            resources.callback(color.release)
            depth = context.depth_renderbuffer(
                (pixel_width, pixel_height), samples=samples
            )
            resources.callback(depth.release)
            frame = context.framebuffer(color_attachments=color, depth_attachment=depth)
        except BaseException:
            try:
                resources.close()
            except BaseException:
                logger.exception("Failed to roll back OpenGL framebuffer allocation")
            raise
        resources.pop_all()
        return frame

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
