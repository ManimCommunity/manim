from __future__ import annotations

import itertools as it
import math
import sys
import time
from typing import Any, Iterable

from manim.renderer.shader_wrapper import ShaderWrapper

from ..constants import RADIANS

if sys.version_info < (3, 8):
    from backports.cached_property import cached_property
else:
    from functools import cached_property

import moderngl
import numpy as np
import OpenGL.GL as gl
from PIL import Image
from scipy.spatial.transform import Rotation

from manim import config, logger
from manim.mobject.opengl.opengl_mobject import OpenGLMobject, OpenGLPoint
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject
from manim.utils.caching import handle_caching_play
from manim.utils.color import BLACK, color_to_rgba
from manim.utils.exceptions import EndSceneEarlyException

from ..constants import *
from ..scene.scene_file_writer import SceneFileWriter
from ..utils import opengl
from ..utils.config_ops import _Data
from ..utils.simple_functions import clip, fdiv
from ..utils.space_ops import (
    angle_of_vector,
    normalize,
    quaternion_from_angle_axis,
    quaternion_mult,
    rotation_matrix_transpose,
    rotation_matrix_transpose_from_quaternion,
)
from .shader import Mesh, Shader
from .vectorized_mobject_rendering import (
    render_opengl_vectorized_mobject_fill,
    render_opengl_vectorized_mobject_stroke,
)


class OpenGLCameraFrame(OpenGLMobject):
    def __init__(
        self,
        frame_shape: tuple[float, float] = (config.frame_width, config.frame_height),
        center_point: np.ndarray = ORIGIN,
        focal_dist_to_height: float = 2.0,
        **kwargs,
    ):
        self.frame_shape = frame_shape
        self.center_point = center_point
        self.focal_dist_to_height = focal_dist_to_height
        super().__init__(**kwargs)

    def init_uniforms(self):
        super().init_uniforms()
        # as a quarternion
        self.uniforms["orientation"] = Rotation.identity().as_quat()
        self.uniforms["focal_dist_to_height"] = self.focal_dist_to_height

    def init_points(self) -> None:
        self.set_points([ORIGIN, LEFT, RIGHT, DOWN, UP])
        self.set_width(self.frame_shape[0], stretch=True)
        self.set_height(self.frame_shape[1], stretch=True)
        self.move_to(self.center_point)

    def set_orientation(self, rotation: Rotation):
        self.uniforms["orientation"] = rotation.as_quat()
        return self

    def get_orientation(self):
        return Rotation.from_quat(self.uniforms["orientation"])

    def to_default_state(self):
        self.center()
        self.set_height(config.frame_width)
        self.set_width(config.frame_height)
        self.set_orientation(Rotation.identity())
        return self

    def get_euler_angles(self):
        return self.get_orientation().as_euler("zxz")[::-1]

    def get_theta(self):
        return self.get_euler_angles()[0]

    def get_phi(self):
        return self.get_euler_angles()[1]

    def get_gamma(self):
        return self.get_euler_angles()[2]

    def get_inverse_camera_rotation_matrix(self):
        return self.get_orientation().as_matrix().T

    def rotate(self, angle: float, axis: np.ndarray = OUT, **kwargs):  # type: ignore
        rot = Rotation.from_rotvec(axis * normalize(axis))  # type: ignore
        self.set_orientation(rot * self.get_orientation())

    def set_euler_angles(
        self,
        theta: float | None = None,
        phi: float | None = None,
        gamma: float | None = None,
        units: float = RADIANS,
    ):
        eulers = self.get_euler_angles()  # theta, phi, gamma
        for i, var in enumerate([theta, phi, gamma]):
            if var is not None:
                eulers[i] = var * units
        self.set_orientation(Rotation.from_euler("zxz", eulers[::-1]))
        return self

    def reorient(
        self,
        theta_degrees: float | None = None,
        phi_degrees: float | None = None,
        gamma_degrees: float | None = None,
    ):
        """
        Shortcut for set_euler_angles, defaulting to taking
        in angles in degrees
        """
        self.set_euler_angles(theta_degrees, phi_degrees, gamma_degrees, units=DEGREES)
        return self

    def set_theta(self, theta: float):
        return self.set_euler_angles(theta=theta)

    def set_phi(self, phi: float):
        return self.set_euler_angles(phi=phi)

    def set_gamma(self, gamma: float):
        return self.set_euler_angles(gamma=gamma)

    def increment_theta(self, dtheta: float):
        self.rotate(dtheta, OUT)
        return self

    def increment_phi(self, dphi: float):
        self.rotate(dphi, self.get_inverse_camera_rotation_matrix()[0])
        return self

    def increment_gamma(self, dgamma: float):
        self.rotate(dgamma, self.get_inverse_camera_rotation_matrix()[2])
        return self

    def set_focal_distance(self, focal_distance: float):
        self.uniforms["focal_dist_to_height"] = focal_distance / self.get_height()
        return self

    def set_field_of_view(self, field_of_view: float):
        self.uniforms["focal_dist_to_height"] = 2 * math.tan(field_of_view / 2)
        return self

    def get_shape(self):
        return (self.get_width(), self.get_height())

    def get_center(self) -> np.ndarray:
        # Assumes first point is at the center
        return self.points[0]

    def get_width(self) -> float:
        points = self.points
        return points[2, 0] - points[1, 0]

    def get_height(self) -> float:
        points = self.points
        return points[4, 1] - points[3, 1]

    def get_focal_distance(self) -> float:
        return self.uniforms["focal_dist_to_height"] * self.get_height()  # type: ignore

    def get_field_of_view(self) -> float:
        return 2 * math.atan(self.uniforms["focal_dist_to_height"] / 2)

    def get_implied_camera_location(self) -> np.ndarray:
        to_camera = self.get_inverse_camera_rotation_matrix()[2]
        dist = self.get_focal_distance()
        return self.get_center() + dist * to_camera


class OpenGLCamera:
    def __init__(
        self,
        ctx: moderngl.Context | None = None,
        background_image: str | None = None,
        frame_config: dict = {},
        pixel_width: int = config.pixel_width,
        pixel_height: int = config.pixel_height,
        fps: int = config.frame_rate,
        # Note: frame height and width will be resized to match the pixel aspect rati
        background_color=BLACK,
        background_opacity: float = 1.0,
        # Points in vectorized mobjects with norm greater
        # than this value will be rescaled
        max_allowable_norm: float = 1.0,
        image_mode: str = "RGBA",
        n_channels: int = 4,
        pixel_array_dtype: type = np.uint8,
        light_source_position: np.ndarray = np.array([-10, 10, 10]),
        # Although vector graphics handle antialiasing fine
        # without multisampling, for 3d scenes one might want
        # to set samples to be greater than 0.
        samples: int = 0,
    ) -> None:
        self.background_image = background_image
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.fps = fps
        self.max_allowable_norm = max_allowable_norm
        self.image_mode = image_mode
        self.n_channels = n_channels
        self.pixel_array_dtype = pixel_array_dtype
        self.light_source_position = light_source_position
        self.samples = samples

        self.rgb_max_val: float = np.iinfo(self.pixel_array_dtype).max
        self.background_color: list[float] = list(
            color_to_rgba(background_color, background_opacity)
        )
        self.init_frame(**frame_config)
        self.init_context(ctx)
        self.init_shaders()
        self.init_textures()
        self.init_light_source()
        self.refresh_perspective_uniforms()
        # A cached map from mobjects to their associated list of render groups
        # so that these render groups are not regenerated unnecessarily for static
        # mobjects
        self.mob_to_render_groups: dict = {}

    def init_frame(self, **config) -> None:
        self.frame = OpenGLCameraFrame(**config)

    def init_context(self, ctx: moderngl.Context | None = None) -> None:
        if ctx is None:
            ctx = moderngl.create_standalone_context()
            fbo = self.get_fbo(ctx, 0)
        else:
            fbo = ctx.detect_framebuffer()

        self.ctx = ctx
        self.fbo = fbo
        self.set_ctx_blending()

        # For multisample antisampling
        fbo_msaa = self.get_fbo(ctx, self.samples)
        fbo_msaa.use()
        self.fbo_msaa = fbo_msaa

    def set_ctx_blending(self, enable: bool = True) -> None:
        if enable:
            self.ctx.enable(moderngl.BLEND)
        else:
            self.ctx.disable(moderngl.BLEND)

    def set_ctx_depth_test(self, enable: bool = True) -> None:
        if enable:
            self.ctx.enable(moderngl.DEPTH_TEST)
        else:
            self.ctx.disable(moderngl.DEPTH_TEST)

    def init_light_source(self) -> None:
        self.light_source = OpenGLPoint(self.light_source_position)

    # Methods associated with the frame buffer
    def get_fbo(self, ctx: moderngl.Context, samples: int = 0) -> moderngl.Framebuffer:
        pw = self.pixel_width
        ph = self.pixel_height
        return ctx.framebuffer(
            color_attachments=ctx.texture(
                (pw, ph), components=self.n_channels, samples=samples
            ),
            depth_attachment=ctx.depth_renderbuffer((pw, ph), samples=samples),
        )

    def clear(self) -> None:
        self.fbo.clear(*self.background_color)
        self.fbo_msaa.clear(*self.background_color)

    def reset_pixel_shape(self, new_width: int, new_height: int) -> None:
        self.pixel_width = new_width
        self.pixel_height = new_height
        self.refresh_perspective_uniforms()

    def get_raw_fbo_data(self, dtype: str = "f1") -> bytes:
        # Copy blocks from the fbo_msaa to the drawn fbo using Blit
        self.ctx.copy_framebuffer(self.fbo_msaa, self.fbo)
        return self.fbo.read(
            viewport=self.fbo.viewport,
            components=self.n_channels,
            dtype=dtype,
        )

    def get_image(self) -> Image.Image:
        return Image.frombytes(
            "RGBA",
            self.get_pixel_shape(),
            self.get_raw_fbo_data(),
            "raw",
            "RGBA",
            0,
            -1,
        )

    def get_pixel_array(self) -> np.ndarray:
        raw = self.get_raw_fbo_data(dtype="f4")
        flat_arr = np.frombuffer(raw, dtype="f4")
        arr = flat_arr.reshape([*reversed(self.fbo.size), self.n_channels])
        arr = arr[::-1]
        # Convert from float
        return (self.rgb_max_val * arr).astype(self.pixel_array_dtype)

    def get_texture(self):
        texture = self.ctx.texture(
            size=self.fbo.size, components=4, data=self.get_raw_fbo_data(), dtype="f4"
        )
        return texture

    # Getting camera attributes
    def get_pixel_shape(self) -> tuple[int, int]:
        return self.fbo.viewport[2:4]
        # return (self.pixel_width, self.pixel_height)

    def get_pixel_width(self) -> int:
        return self.get_pixel_shape()[0]

    def get_pixel_height(self) -> int:
        return self.get_pixel_shape()[1]

    def get_frame_height(self) -> float:
        return self.frame.get_height()

    def get_frame_width(self) -> float:
        return self.frame.get_width()

    def get_frame_shape(self) -> tuple[float, float]:
        return (self.get_frame_width(), self.get_frame_height())

    def get_frame_center(self) -> np.ndarray:
        return self.frame.get_center()

    def get_location(self) -> tuple[float, float, float] | np.ndarray:
        return self.frame.get_implied_camera_location()

    def resize_frame_shape(self, fixed_dimension: bool = False) -> None:
        """
        Changes frame_shape to match the aspect ratio
        of the pixels, where fixed_dimension determines
        whether frame_height or frame_width
        remains fixed while the other changes accordingly.
        """
        pixel_height = self.get_pixel_height()
        pixel_width = self.get_pixel_width()
        frame_height = self.get_frame_height()
        frame_width = self.get_frame_width()
        aspect_ratio = fdiv(pixel_width, pixel_height)
        if not fixed_dimension:
            frame_height = frame_width / aspect_ratio
        else:
            frame_width = aspect_ratio * frame_height
        self.frame.set_height(frame_height)
        self.frame.set_width(frame_width)

    # Rendering
    def capture(self, *mobjects: OpenGLMobject) -> None:
        self.refresh_perspective_uniforms()
        for mobject in mobjects:
            for render_group in self.get_render_group_list(mobject):
                self.render(render_group)

    def render(self, render_group: dict[str, Any]) -> None:
        shader_wrapper: ShaderWrapper = render_group["shader_wrapper"]
        shader_program = render_group["prog"]
        self.set_shader_uniforms(shader_program, shader_wrapper)
        self.set_ctx_depth_test(shader_wrapper.depth_test)
        render_group["vao"].render(int(shader_wrapper.render_primitive))
        if render_group["single_use"]:
            self.release_render_group(render_group)

    def get_render_group_list(self, mobject: OpenGLMobject) -> Iterable[dict[str, Any]]:
        if mobject.is_changing():
            return self.generate_render_group_list(mobject)

        # Otherwise, cache result for later use
        key = id(mobject)
        if key not in self.mob_to_render_groups:
            self.mob_to_render_groups[key] = list(
                self.generate_render_group_list(mobject)
            )
        return self.mob_to_render_groups[key]

    def generate_render_group_list(
        self, mobject: OpenGLMobject
    ) -> Iterable[dict[str, Any]]:
        return (
            self.get_render_group(sw, single_use=mobject.is_changing())
            for sw in mobject.get_shader_wrapper_list()
        )

    def get_render_group(
        self, shader_wrapper: ShaderWrapper, single_use: bool = True
    ) -> dict[str, Any]:
        # Data buffers
        vbo = self.ctx.buffer(shader_wrapper.vert_data.tobytes())
        if shader_wrapper.vert_indices is None:
            ibo = None
        else:
            vert_index_data = shader_wrapper.vert_indices.astype("i4").tobytes()
            if vert_index_data:
                ibo = self.ctx.buffer(vert_index_data)
            else:
                ibo = None

        # Program an vertex array
        shader_program, vert_format = self.get_shader_program(shader_wrapper)  # type: ignore
        vao = self.ctx.vertex_array(
            program=shader_program,
            content=[(vbo, vert_format, *shader_wrapper.vert_attributes)],
            index_buffer=ibo,
        )
        return {
            "vbo": vbo,
            "ibo": ibo,
            "vao": vao,
            "prog": shader_program,
            "shader_wrapper": shader_wrapper,
            "single_use": single_use,
        }

    def release_render_group(self, render_group: dict[str, Any]) -> None:
        for key in ["vbo", "ibo", "vao"]:
            if render_group[key] is not None:
                render_group[key].release()

    def refresh_static_mobjects(self) -> None:
        for render_group in it.chain(*self.mob_to_render_groups.values()):
            self.release_render_group(render_group)
        self.mob_to_render_groups = {}

    # Shaders
    def init_shaders(self) -> None:
        # Initialize with the null id going to None
        self.id_to_shader_program: dict[int, tuple[moderngl.Program, str] | None] = {
            hash(""): None
        }

    def get_shader_program(
        self, shader_wrapper: ShaderWrapper
    ) -> tuple[moderngl.Program, str] | None:
        sid = shader_wrapper.get_program_id()
        if sid not in self.id_to_shader_program:
            # Create shader program for the first time, then cache
            # in the id_to_shader_program dictionary
            program = self.ctx.program(**shader_wrapper.get_program_code())
            vert_format = moderngl.detect_format(
                program, shader_wrapper.vert_attributes
            )
            self.id_to_shader_program[sid] = (program, vert_format)

        return self.id_to_shader_program[sid]

    def set_shader_uniforms(
        self,
        shader: moderngl.Program,
        shader_wrapper: ShaderWrapper,
    ) -> None:
        for name, path in shader_wrapper.texture_paths.items():
            tid = self.get_texture_id(path)
            shader[name].value = tid
        for name, value in it.chain(
            self.perspective_uniforms.items(), shader_wrapper.uniforms.items()
        ):
            if name in shader:
                if isinstance(value, np.ndarray) and value.ndim > 0:
                    value = tuple(value)
                shader[name].value = value

    def refresh_perspective_uniforms(self) -> None:
        frame = self.frame
        # Orient light
        rotation = frame.get_inverse_camera_rotation_matrix()
        offset = frame.get_center()
        light_pos = np.dot(rotation, self.light_source.get_location() + offset)
        cam_pos = self.frame.get_implied_camera_location()  # TODO

        self.perspective_uniforms = {
            "frame_shape": frame.get_shape(),
            "pixel_shape": self.get_pixel_shape(),
            "camera_offset": tuple(offset),
            "camera_rotation": tuple(np.array(rotation).T.flatten()),
            "camera_position": tuple(cam_pos),
            "light_source_position": tuple(light_pos),
            "focal_distance": frame.get_focal_distance(),
        }

    def init_textures(self) -> None:
        self.n_textures: int = 0
        self.path_to_texture: dict[str, tuple[int, moderngl.Texture]] = {}

    def get_texture_id(self, path: str) -> int:
        if path not in self.path_to_texture:
            if self.n_textures == 15:  # I have no clue why this is needed
                self.n_textures += 1
            tid = self.n_textures
            self.n_textures += 1
            im = Image.open(path).convert("RGBA")
            texture = self.ctx.texture(
                size=im.size,
                components=len(im.getbands()),
                data=im.tobytes(),
            )
            texture.use(location=tid)
            self.path_to_texture[path] = (tid, texture)
        return self.path_to_texture[path][0]

    def release_texture(self, path: str):
        tid_and_texture = self.path_to_texture.pop(path, None)
        if tid_and_texture:
            tid_and_texture[1].release()
        return self


class OpenGLCameraLegacy(OpenGLMobject):
    euler_angles = _Data()

    def __init__(
        self,
        frame_shape=None,
        center_point=None,
        # Theta, phi, gamma
        euler_angles=[0, 0, 0],
        focal_distance=2,
        light_source_position=[-10, 10, 10],
        orthographic=False,
        minimum_polar_angle=-PI / 2,
        maximum_polar_angle=PI / 2,
        model_matrix=None,
        **kwargs,
    ):
        self.use_z_index = True
        self.frame_rate = 60
        self.orthographic = orthographic
        self.minimum_polar_angle = minimum_polar_angle
        self.maximum_polar_angle = maximum_polar_angle
        if self.orthographic:
            self.projection_matrix = opengl.orthographic_projection_matrix()
            self.unformatted_projection_matrix = opengl.orthographic_projection_matrix(
                format=False,
            )
        else:
            self.projection_matrix = opengl.perspective_projection_matrix()
            self.unformatted_projection_matrix = opengl.perspective_projection_matrix(
                format=False,
            )

        if frame_shape is None:
            self.frame_shape = (config["frame_width"], config["frame_height"])
        else:
            self.frame_shape = frame_shape

        if center_point is None:
            self.center_point = ORIGIN
        else:
            self.center_point = center_point

        if model_matrix is None:
            model_matrix = opengl.translation_matrix(0, 0, 11)

        self.focal_distance = focal_distance

        if light_source_position is None:
            self.light_source_position = [-10, 10, 10]
        else:
            self.light_source_position = light_source_position
        self.light_source = OpenGLPoint(self.light_source_position)

        self.default_model_matrix = model_matrix
        super().__init__(model_matrix=model_matrix, should_render=False, **kwargs)

        if euler_angles is None:
            euler_angles = [0, 0, 0]
        euler_angles = np.array(euler_angles, dtype=float)

        self.euler_angles = euler_angles
        self.refresh_rotation_matrix()

    def get_position(self):
        return self.model_matrix[:, 3][:3]

    def set_position(self, position):
        self.model_matrix[:, 3][:3] = position
        return self

    @cached_property
    def formatted_view_matrix(self):
        return opengl.matrix_to_shader_input(np.linalg.inv(self.model_matrix))

    @cached_property
    def unformatted_view_matrix(self):
        return np.linalg.inv(self.model_matrix)

    def init_points(self):
        self.set_points([ORIGIN, LEFT, RIGHT, DOWN, UP])
        self.set_width(self.frame_shape[0], stretch=True)
        self.set_height(self.frame_shape[1], stretch=True)
        self.move_to(self.center_point)

    def to_default_state(self):
        self.center()
        self.set_height(config["frame_height"])
        self.set_width(config["frame_width"])
        self.set_euler_angles(0, 0, 0)
        self.model_matrix = self.default_model_matrix
        return self

    def refresh_rotation_matrix(self):
        # Rotate based on camera orientation
        theta, phi, gamma = self.euler_angles
        quat = quaternion_mult(
            quaternion_from_angle_axis(theta, OUT, axis_normalized=True),
            quaternion_from_angle_axis(phi, RIGHT, axis_normalized=True),
            quaternion_from_angle_axis(gamma, OUT, axis_normalized=True),
        )
        self.inverse_rotation_matrix = rotation_matrix_transpose_from_quaternion(quat)

    def rotate(self, angle, axis=OUT, **kwargs):
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

    def set_euler_angles(self, theta=None, phi=None, gamma=None):
        if theta is not None:
            self.euler_angles[0] = theta
        if phi is not None:
            self.euler_angles[1] = phi
        if gamma is not None:
            self.euler_angles[2] = gamma
        self.refresh_rotation_matrix()
        return self

    def set_theta(self, theta):
        return self.set_euler_angles(theta=theta)

    def set_phi(self, phi):
        return self.set_euler_angles(phi=phi)

    def set_gamma(self, gamma):
        return self.set_euler_angles(gamma=gamma)

    def increment_theta(self, dtheta):
        self.euler_angles[0] += dtheta
        self.refresh_rotation_matrix()
        return self

    def increment_phi(self, dphi):
        phi = self.euler_angles[1]
        new_phi = clip(phi + dphi, -PI / 2, PI / 2)
        self.euler_angles[1] = new_phi
        self.refresh_rotation_matrix()
        return self

    def increment_gamma(self, dgamma):
        self.euler_angles[2] += dgamma
        self.refresh_rotation_matrix()
        return self

    def get_shape(self):
        return (self.get_width(), self.get_height())

    def get_center(self):
        # Assumes first point is at the center
        return self.points[0]

    def get_width(self):
        points = self.points
        return points[2, 0] - points[1, 0]

    def get_height(self):
        points = self.points
        return points[4, 1] - points[3, 1]

    def get_focal_distance(self):
        return self.focal_distance * self.get_height()

    def interpolate(self, *args, **kwargs):
        super().interpolate(*args, **kwargs)
        self.refresh_rotation_matrix()


points_per_curve = 3


class OpenGLRenderer:
    def __init__(self, file_writer_class=SceneFileWriter, skip_animations=False):
        # Measured in pixel widths, used for vector graphics
        self.anti_alias_width = 1.5
        self._file_writer_class = file_writer_class

        self._original_skipping_status = skip_animations
        self.skip_animations = skip_animations
        self.animation_start_time = 0
        self.animation_elapsed_time = 0
        self.time = 0
        self.animations_hashes = []
        self.num_plays = 0

        self.camera = OpenGLCamera()
        self.pressed_keys = set()

        # Initialize texture map.
        self.path_to_texture_id = {}

        self.background_color = config["background_color"]

    def init_scene(self, scene):
        self.partial_movie_files = []
        self.file_writer: Any = self._file_writer_class(
            self,
            scene.__class__.__name__,
        )
        self.scene = scene
        self.background_color = config["background_color"]
        if not hasattr(self, "window"):
            if self.should_create_window():
                from .opengl_renderer_window import Window

                self.window = Window(self)
                self.context = self.window.ctx
                self.frame_buffer_object = self.context.detect_framebuffer()
            else:
                self.window = None
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

    def should_create_window(self):
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

    def get_pixel_shape(self):
        if hasattr(self, "frame_buffer_object"):
            return self.frame_buffer_object.viewport[2:4]
        else:
            return None

    def refresh_perspective_uniforms(self, camera):
        pw, ph = self.get_pixel_shape()
        fw, fh = camera.get_shape()
        # TODO, this should probably be a mobject uniform, with
        # the camera taking care of the conversion factor
        anti_alias_width = self.anti_alias_width / (ph / fh)
        # Orient light
        rotation = camera.inverse_rotation_matrix
        light_pos = camera.light_source.get_location()
        light_pos = np.dot(rotation, light_pos)

        self.perspective_uniforms = {
            "frame_shape": camera.get_shape(),
            "anti_alias_width": anti_alias_width,
            "camera_center": tuple(camera.get_center()),
            "camera_rotation": tuple(np.array(rotation).T.flatten()),
            "light_source_position": tuple(light_pos),
            "focal_distance": camera.get_focal_distance(),
        }

    def render_mobject(self, mobject):
        if isinstance(mobject, OpenGLVMobject):
            if config["use_projection_fill_shaders"]:
                render_opengl_vectorized_mobject_fill(self, mobject)

            if config["use_projection_stroke_shaders"]:
                render_opengl_vectorized_mobject_stroke(self, mobject)

        shader_wrapper_list = mobject.get_shader_wrapper_list()

        # Convert ShaderWrappers to Meshes.
        for shader_wrapper in shader_wrapper_list:
            shader = Shader(self.context, shader_wrapper.shader_folder)

            # Set textures.
            for name, path in shader_wrapper.texture_paths.items():
                tid = self.get_texture_id(path)
                shader.shader_program[name].value = tid

            # Set uniforms.
            for name, value in it.chain(
                shader_wrapper.uniforms.items(),
                self.perspective_uniforms.items(),
            ):
                try:
                    shader.set_uniform(name, value)
                except KeyError:
                    pass
            try:
                shader.set_uniform(
                    "u_view_matrix", self.scene.camera.formatted_view_matrix
                )
                shader.set_uniform(
                    "u_projection_matrix",
                    self.scene.camera.projection_matrix,
                )
            except KeyError:
                pass

            # Set depth test.
            if shader_wrapper.depth_test:
                self.context.enable(moderngl.DEPTH_TEST)
            else:
                self.context.disable(moderngl.DEPTH_TEST)

            # Render.
            mesh = Mesh(
                shader,
                shader_wrapper.vert_data,
                indices=shader_wrapper.vert_indices,
                use_depth_test=shader_wrapper.depth_test,
                primitive=mobject.render_primitive,
            )
            mesh.set_uniforms(self)
            mesh.render()

    def get_texture_id(self, path):
        if repr(path) not in self.path_to_texture_id:
            tid = len(self.path_to_texture_id)
            texture = self.context.texture(
                size=path.size,
                components=len(path.getbands()),
                data=path.tobytes(),
            )
            texture.repeat_x = False
            texture.repeat_y = False
            texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            texture.swizzle = "RRR1" if path.mode == "L" else "RGBA"
            texture.use(location=tid)
            self.path_to_texture_id[repr(path)] = tid

        return self.path_to_texture_id[repr(path)]

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

    @handle_caching_play
    def play(self, scene, *args, **kwargs):
        # TODO: Handle data locking / unlocking.
        self.animation_start_time = time.time()
        self.file_writer.begin_animation(not self.skip_animations)

        scene.compile_animation_data(*args, **kwargs)
        scene.begin_animations()
        if scene.is_current_animation_frozen_frame():
            self.update_frame(scene)

            if not self.skip_animations:
                for _ in range(int(config.frame_rate * scene.duration)):
                    self.file_writer.write_frame(self)

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

    def clear_screen(self):
        self.frame_buffer_object.clear(*self.background_color)
        self.window.swap_buffers()

    def render(self, scene, frame_offset, moving_mobjects):
        self.update_frame(scene)

        if self.skip_animations:
            return

        self.file_writer.write_frame(self)

        if self.window is not None:
            self.window.swap_buffers()
            while self.animation_elapsed_time < frame_offset:
                self.update_frame(scene)
                self.window.swap_buffers()

    def update_frame(self, scene):
        self.frame_buffer_object.clear(*self.background_color)
        self.refresh_perspective_uniforms(scene.camera)

        for mobject in scene.mobjects:
            self.render_mobject(mobject)

        for obj in scene.meshes:
            for mesh in obj.get_meshes():
                mesh.set_uniforms(self)
                mesh.render()

        self.animation_elapsed_time = time.time() - self.animation_start_time

    def scene_finished(self, scene):
        # When num_plays is 0, no images have been output, so output a single
        # image in this case
        if self.num_plays > 0:
            self.file_writer.finish()
        elif self.num_plays == 0 and config.write_to_movie:
            config.write_to_movie = False

        if self.should_save_last_frame():
            config.save_last_frame = True
            self.update_frame(scene)
            self.file_writer.save_final_image(self.get_image())

    def should_save_last_frame(self):
        if config["save_last_frame"]:
            return True
        if self.scene.interactive_mode:
            return False
        return self.num_plays == 0

    def get_image(self) -> Image.Image:
        """Returns an image from the current frame. The first argument passed to image represents
        the mode RGB with the alpha channel A. The data we read is from the currently bound frame
        buffer. We pass in 'raw' as the name of the decoder, 0 and -1 args are specifically
        used for the decoder tand represent the stride and orientation. 0 means there is no
        padding expected between bytes and -1 represents the orientation and means the first
        line of the image is the bottom line on the screen.

        Returns
        -------
        PIL.Image
            The PIL image of the array.
        """
        raw_buffer_data = self.get_raw_frame_buffer_object_data()
        image = Image.frombytes(
            "RGBA",
            self.get_pixel_shape(),
            raw_buffer_data,
            "raw",
            "RGBA",
            0,
            -1,
        )
        return image

    def save_static_frame_data(self, scene, static_mobjects):
        pass

    def get_frame_buffer_object(self, context, samples=0):
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

    def get_raw_frame_buffer_object_data(self, dtype="f1"):
        # Copy blocks from the fbo_msaa to the drawn fbo using Blit
        # pw, ph = self.get_pixel_shape()
        # gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.fbo_msaa.glo)
        # gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self.fbo.glo)
        # gl.glBlitFramebuffer(
        #     0, 0, pw, ph, 0, 0, pw, ph, gl.GL_COLOR_BUFFER_BIT, gl.GL_LINEAR
        # )
        num_channels = 4
        ret = self.frame_buffer_object.read(
            viewport=self.frame_buffer_object.viewport,
            components=num_channels,
            dtype=dtype,
        )
        return ret

    def get_frame(self):
        # get current pixel values as numpy data in order to test output
        raw = self.get_raw_frame_buffer_object_data(dtype="f1")
        pixel_shape = self.get_pixel_shape()
        result_dimensions = (pixel_shape[1], pixel_shape[0], 4)
        np_buf = np.frombuffer(raw, dtype="uint8").reshape(result_dimensions)
        np_buf = np.flipud(np_buf)
        return np_buf

    # Returns offset from the bottom left corner in pixels.
    # top_left flag should be set to True when using a GUI framework
    # where the (0,0) is at the top left: e.g. PySide6
    def pixel_coords_to_space_coords(self, px, py, relative=False, top_left=False):
        pixel_shape = self.get_pixel_shape()
        if pixel_shape is None:
            return np.array([0, 0, 0])
        pw, ph = pixel_shape
        fw, fh = config["frame_width"], config["frame_height"]
        fc = self.camera.get_center()
        if relative:
            return 2 * np.array([px / pw, py / ph, 0])
        else:
            # Only scale wrt one axis
            scale = fh / ph
            return fc + scale * np.array(
                [(px - pw / 2), (-1 if top_left else 1) * (py - ph / 2), 0]
            )

    @property
    def background_color(self):
        return self._background_color

    @background_color.setter
    def background_color(self, value):
        self._background_color = color_to_rgba(value, 1.0)
