from __future__ import annotations

import contextlib
import itertools as it
import time
import typing
from functools import cached_property
from typing import TYPE_CHECKING, Any, Self

import moderngl
import numpy as np
from moderngl import Framebuffer
from PIL import Image
from typing_extensions import override

from manim import config, logger
from manim.mobject.opengl.opengl_mobject import (
    OpenGLMobject,
    OpenGLPoint,
)
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject
from manim.utils.caching import handle_caching_play
from manim.utils.color import color_to_rgba
from manim.utils.exceptions import EndSceneEarlyException
from manim.utils.paths import straight_path

from ..constants import *
from ..scene.scene_file_writer import SceneFileWriter
from ..utils import opengl
from ..utils.simple_functions import clip
from ..utils.space_ops import (
    angle_of_vector,
    quaternion_from_angle_axis,
    quaternion_mult,
    rotation_matrix_transpose,
    rotation_matrix_transpose_from_quaternion,
)
from .opengl_renderer_window import Window
from .shader import Mesh, Shader
from .vectorized_mobject_rendering import (
    render_opengl_vectorized_mobject_fill,
    render_opengl_vectorized_mobject_stroke,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Self

    from manim.animation.animation import Animation
    from manim.mobject.mobject import Mobject, _AnimationBuilder
    from manim.scene.scene import Scene
    from manim.typing import (
        FloatRGBA,
        MatrixMN,
        PathFuncType,
        Point3D,
        Point3DLike,
        RGBAPixelArray,
        Vector3DLike,
    )
    from manim.utils.color.core import ParsableManimColor
    from manim.utils.opengl import FlattenedMatrix4x4


__all__ = ["OpenGLCamera", "OpenGLRenderer"]


class OpenGLCamera(OpenGLMobject):
    """
    An OpenGL-based camera for 3D scene rendering.


    Attributes
    ----------
    frame_shape : tuple[float, float]
        The width and height of the camera frame.
    center_point : np.ndarray
        The center point of the camera in 3D space.
    euler_angles : np.ndarray
        The Euler angles (theta, phi, gamma) representing the camera's orientation.
    focal_distance : float
        The focal distance of the camera.
    light_source_position : np.ndarray
        The position of the light source in 3D space.
    orthographic : bool
        Whether the camera uses orthographic projection instead of perspective.
    minimum_polar_angle : float
        The minimum polar angle for camera rotation.
    maximum_polar_angle : float
        The maximum polar angle for camera rotation.
    inverse_rotation_matrix : np.ndarray
        The inverse rotation matrix of the camera.
    """

    def __init__(
        self,
        frame_shape: tuple[float, float] | None = None,
        center_point: Point3DLike | None = None,
        # Theta, phi, gamma
        euler_angles: Point3DLike | None = None,
        focal_distance: float = 2.0,
        light_source_position: Point3DLike | None = None,
        orthographic: bool = False,
        minimum_polar_angle: float = -PI / 2,
        maximum_polar_angle: float = PI / 2,
        model_matrix: MatrixMN | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes an OpenGLCamera instance.

        Parameters
        ----------
        frame_shape : tuple[float, float], optional
            The width and height of the camera frame. If not provided, defaults to
            the global manim config values `frame_width` and `frame_height`.
        center_point : Point3DLike, optional
            The center point of the camera in 3D space.
            If not provided, defaults to the origin (0, 0, 0).
        euler_angles : Point3DLike, optional
            The Euler angles (theta, phi, gamma) representing the camera's orientation.
            If not provided, defaults to (0, 0, 0) (i.e., no rotation).
        focal_distance : float, optional
            The focal distance of the camera. Default is 2.0.
        light_source_position : Point3DLike, optional
            The position of the light source in 3D space.
            If not provided, defaults to (-10, 10, 10).
        orthographic : bool, optional
            Whether the camera uses orthographic projection instead of perspective.
            Default is False (perspective).
        minimum_polar_angle : float, optional
            The minimum polar angle in radian for camera rotation. Default is -π/2,
            i.e. no restriction.
        maximum_polar_angle : float, optional
            The maximum polar angle in radian for camera rotation. Default is π/2,
            i.e. no restriction.
        model_matrix : MatrixMN, optional
            The initial model matrix [1]_ for the camera. If not provided,
            defaults to a translation matrix that positions the camera at (0, 0, 11).
        **kwargs : Any
            Additional keyword arguments passed to the OpenGLMobject constructor.

        References
        ----------
        .. [1] Wikipedia, "Camera matrix",
               https://en.wikipedia.org/wiki/Camera_matrix
        """
        self.use_z_index = True
        self.frame_rate = 60
        self.orthographic = orthographic
        self.minimum_polar_angle = minimum_polar_angle
        self.maximum_polar_angle = maximum_polar_angle
        if self.orthographic:
            self.projection_matrix = opengl.orthographic_projection_matrix()
            self.unformatted_projection_matrix = opengl.orthographic_projection_matrix(
                format_=False,
            )
        else:
            self.projection_matrix = opengl.perspective_projection_matrix()
            self.unformatted_projection_matrix = opengl.perspective_projection_matrix(
                format_=False,
            )

        if frame_shape is None:
            self.frame_shape = (config["frame_width"], config["frame_height"])
        else:
            self.frame_shape = frame_shape

        if center_point is None:
            self.center_point = ORIGIN
        else:
            self.center_point = np.asarray(center_point, dtype=float)

        if model_matrix is None:
            model_matrix = opengl.translation_matrix(0, 0, 11)

        self.focal_distance = focal_distance

        self.light_source_position = np.asarray(
            light_source_position or [-10, 10, 10], dtype=float
        )

        self.light_source = OpenGLPoint(self.light_source_position)

        self.default_model_matrix = model_matrix
        super().__init__(model_matrix=model_matrix, should_render=False, **kwargs)

        euler_angles = np.asarray(euler_angles or [0, 0, 0], dtype=float)

        self.euler_angles: Point3D = euler_angles
        self.refresh_rotation_matrix()

    def get_position(self) -> Point3D:
        """Retrieve the camera's position in 3D space."""
        return self.model_matrix[:, 3][:3]

    def set_position(self, position: Point3D) -> Self:
        """Set the camera's position in 3D space."""
        self.model_matrix[:, 3][:3] = position
        return self

    @cached_property
    def formatted_view_matrix(self) -> FlattenedMatrix4x4:
        """The formatted view matrix for shader input."""
        return opengl.matrix_to_shader_input(self.unformatted_view_matrix)

    @cached_property
    def unformatted_view_matrix(self) -> MatrixMN:
        return np.linalg.inv(self.model_matrix)

    def init_points(self) -> None:
        """Initialize the camera's points based on frame shape and center point."""
        self.set_points([ORIGIN, LEFT, RIGHT, DOWN, UP])
        self.set_width(self.frame_shape[0], stretch=True)
        self.set_height(self.frame_shape[1], stretch=True)
        self.move_to(self.center_point)

    def to_default_state(self) -> Self:
        """Reset the camera to its default state
        (config frame size, centered at origin, no rotation).
        """
        self.center()
        self.set_height(config["frame_height"])
        self.set_width(config["frame_width"])
        self.set_euler_angles(0, 0, 0)
        self.model_matrix = self.default_model_matrix
        return self

    def refresh_rotation_matrix(self) -> None:
        """Refresh the camera's inverse rotation matrix based on its Euler angles."""
        # Rotate based on camera orientation
        theta, phi, gamma = self.euler_angles
        quat = quaternion_mult(
            quaternion_from_angle_axis(theta, OUT, axis_normalized=True),
            quaternion_from_angle_axis(phi, RIGHT, axis_normalized=True),
            quaternion_from_angle_axis(gamma, OUT, axis_normalized=True),
        )
        self.inverse_rotation_matrix = rotation_matrix_transpose_from_quaternion(quat)

    @override
    def rotate(
        self,
        angle: float,
        axis: Vector3DLike = OUT,
        about_point: Point3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Rotate the camera by a given angle around a specified axis.

        Parameters
        ----------
        angle : float
            The angle in radians to rotate the camera.
        axis : Vector3DLike, optional
            The axis around which to rotate the camera. Default is OUT (z-axis).
        about_point : Point3DLike, optional
            Ignored. For OpenGLCamera, rotation is always about the camera's center.

        **kwargs : Any
            Not used for OpenGLCamera. Passing additional keyword arguments
            has no effect.

        Returns
        -------
        Self
            The rotated camera instance. Returned for chaining.
        """
        curr_rot_T = self.inverse_rotation_matrix
        added_rot_T = rotation_matrix_transpose(angle, axis)
        new_rot_T = np.dot(curr_rot_T, added_rot_T)
        Fz = new_rot_T[2]
        phi = np.arccos(Fz[2])
        theta = angle_of_vector(Fz[:2]) + PI / 2
        partial_rot_T = np.dot(
            rotation_matrix_transpose(phi, RIGHT),
            rotation_matrix_transpose(theta, OUT),
        )
        gamma = angle_of_vector(np.dot(partial_rot_T, new_rot_T.T)[:, 0])
        self.set_euler_angles(theta, phi, gamma)
        return self

    def set_euler_angles(
        self,
        theta: float | None = None,
        phi: float | None = None,
        gamma: float | None = None,
    ) -> Self:
        """
        Set the camera's Euler angles [1]_ (theta, phi, gamma).

        Parameters
        ----------
        theta : float | None, optional
            The angle in radians for rotation around the OUT (z) axis.
            If None, the current theta value is retained.
        phi : float | None, optional
            The angle in radians for rotation around the RIGHT (x) axis.
            If None, the current phi value is retained.
        gamma : float | None, optional
            The angle in radians for rotation around the OUT (z) axis.
            If None, the current gamma value is retained.

        Returns
        -------
        Self
            The camera instance with updated Euler angles. Returned for chaining.

        See Also
        --------
        set_theta : Set the theta Euler angle.
        set_phi : Set the phi Euler angle.
        set_gamma : Set the gamma Euler angle.

        References
        ----------
        .. [1] Wikipedia, "Euler angles",
               https://en.wikipedia.org/wiki/Euler_angles
        """
        if theta is not None:
            self.euler_angles[0] = theta
        if phi is not None:
            self.euler_angles[1] = phi
        if gamma is not None:
            self.euler_angles[2] = gamma
        self.refresh_rotation_matrix()
        return self

    def set_theta(self, theta: float) -> Self:
        """
        Set the camera's theta Euler angle (in radians).

        See Also
        --------
        set_euler_angles : Set all Euler angles at once.
        set_phi : Set the phi Euler angle.
        set_gamma : Set the gamma Euler angle.
        """
        return self.set_euler_angles(theta=theta)

    def set_phi(self, phi: float) -> Self:
        """
        Set the camera's phi Euler angle (in radians).

        See Also
        --------
        set_euler_angles : Set all Euler angles at once.
        set_theta : Set the theta Euler angle.
        set_gamma : Set the gamma Euler angle.
        """
        return self.set_euler_angles(phi=phi)

    def set_gamma(self, gamma: float) -> Self:
        """
        Set the camera's gamma Euler angle (in radians).

        See Also
        --------
        set_euler_angles : Set all Euler angles at once.
        set_theta : Set the theta Euler angle.
        set_phi : Set the phi Euler angle.
        """
        return self.set_euler_angles(gamma=gamma)

    def increment_theta(self, dtheta: float) -> Self:
        """
        Increment the camera's theta Euler angle by a given amount (in radians).

        See Also
        --------
        set_euler_angles : Set all Euler angles at once.
        set_theta : Set the theta Euler angle.
        """
        self.euler_angles[0] += dtheta
        self.refresh_rotation_matrix()
        return self

    def increment_phi(self, dphi: float) -> Self:
        """
        Increment the camera's phi Euler angle by a given amount (in radians).

        See Also
        --------
        set_euler_angles : Set all Euler angles at once.
        set_phi : Set the phi Euler angle.
        """
        phi = self.euler_angles[1]
        new_phi = clip(phi + dphi, -PI / 2, PI / 2)
        self.euler_angles[1] = new_phi
        self.refresh_rotation_matrix()
        return self

    def increment_gamma(self, dgamma: float) -> Self:
        """
        Increment the camera's gamma Euler angle by a given amount (in radians).

        See Also
        --------
        set_euler_angles : Set all Euler angles at once.
        set_gamma : Set the gamma Euler angle.
        """
        self.euler_angles[2] += dgamma
        self.refresh_rotation_matrix()
        return self

    def get_shape(self) -> tuple[float, float]:
        """Retrieve the width and height of the camera frame."""
        return (self.get_width(), self.get_height())

    def get_center(self) -> Point3D:
        """
        Retrieve the center point of the camera in 3D space.

        Notes
        -----
        The center point is assumed to be the first point in the camera's points array.
        """
        # Assumes first point is at the center
        return self.points[0]

    def get_width(self) -> float:
        """Retrieve the width of the camera frame."""
        points = self.points
        out = points[2, 0] - points[1, 0]
        return float(out)

    def get_height(self) -> float:
        """Retrieve the height of the camera frame."""
        points = self.points
        out = points[4, 1] - points[3, 1]
        return float(out)
        # return points[4, 1] - points[3, 1]

    def get_focal_distance(self) -> float:
        """Retrieve the focal distance of the camera."""
        return self.focal_distance * self.get_height()

    @override
    def interpolate(
        self,
        mobject1: OpenGLMobject,
        mobject2: OpenGLMobject,
        alpha: float,
        path_func: PathFuncType = straight_path(),
    ):
        super().interpolate(mobject1, mobject2, alpha, path_func)
        self.refresh_rotation_matrix()
        return self


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

    def init_scene(self, scene: Scene) -> None:
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
            self,
            scene.__class__.__name__,
        )
        self.scene = scene

        self.background_color = config["background_color"]
        if self.should_create_window():
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
        self.context.enable(moderngl.BLEND)
        self.context.wireframe = config["enable_wireframe"]
        self.context.blend_func = (
            moderngl.SRC_ALPHA,
            moderngl.ONE_MINUS_SRC_ALPHA,
            moderngl.ONE,
            moderngl.ONE,
        )

    def should_create_window(self) -> bool:
        """
        Determine whether a window should be created for rendering
        based on the current configuration.

        Notes
        -----
        A windows is always created if the 'force_window' configuration is enabled.
        """
        if config["force_window"]:
            logger.warning(
                "'--force_window' is enabled, this is intended for debugging purposes "
                "and may impact performance if used when outputting files",
            )
            return True
        return (
            config["preview"]
            and not config["save_last_frame"]
            and not config["format"]
            and not config["write_to_movie"]
            and not config["dry_run"]
        )

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
        self.file_writer.begin_animation(not self.skip_animations)

        scene.compile_animation_data(*animations, **kwargs)
        scene.begin_animations()
        if scene.is_current_animation_frozen_frame():
            self.update_frame(scene)

            if not self.skip_animations:
                self.file_writer.write_frame(
                    self, num_frames=int(config.frame_rate * scene.duration)
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

        self.file_writer.write_frame(self)

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

        self.animation_elapsed_time = time.time() - self.animation_start_time

    def scene_finished(self, scene: Scene) -> None:
        """
        Handle the finalization process after a scene has finished rendering.

        Performs the following actions:
        - If any plays (animations) have occurred, finalizes the file writing process.
        - If no plays have occurred but movie writing is enabled, disables
          movie writing to avoid creating an empty movie file.
        - If the configuration requires saving the last frame,
          updates and saves the final image of the scene.

        Parameters
        ----------
        scene : Scene
            The scene that has finished rendering.
        """
        # When num_plays is 0, no images have been output, so output a single
        # image in this case
        if self.num_plays > 0:
            self.file_writer.finish()
        elif self.num_plays == 0 and config.write_to_movie:
            config.write_to_movie = False

        if self.should_save_last_frame():
            config.save_last_frame = True
            self.update_frame(scene)
            self.file_writer.save_image(self.get_image())

    def should_save_last_frame(self) -> bool:
        """
        Determine whether the last frame of the scene should be saved,
        i.e. if one of the following conditions is met:
        - The configuration option 'save_last_frame' is enabled.
        - The scene is not in interactive mode.
        - This is the first play (i.e., num_plays == 0).
        """
        if config["save_last_frame"]:
            return True
        if self.scene.interactive_mode:
            return False
        return self.num_plays == 0

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
        np_buf = np.flipud(np_buf)
        return np_buf

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
            return np.array([0, 0, 0])
        pixel_width, pixel_height = pixel_shape
        frame_height = config["frame_height"]
        frame_center = self.camera.get_center()
        if relative:
            # relative -> just normalize to [-1, 1]
            return 2 * np.array([px / pixel_width, py / pixel_height, 0])

        scale = frame_height / pixel_height
        y_direction = -1 if top_left else 1

        return frame_center + scale * np.array(
            [
                (px - pixel_width / 2),
                y_direction * (py - pixel_height / 2),
                0,
            ]
        )

    @property
    def background_color(self) -> FloatRGBA:
        """The background color of the renderer (RGBA format)."""
        return self._background_color

    @background_color.setter
    def background_color(self, value: ParsableManimColor) -> None:
        self._background_color = color_to_rgba(value, 1.0)
