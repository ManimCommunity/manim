from __future__ import annotations

import contextlib
import itertools as it
import time
from functools import cached_property
from typing import Any

import moderngl
import numpy as np
from PIL import Image

from manim import config, logger
from manim.mobject.opengl.opengl_mobject import OpenGLMobject, OpenGLPoint
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject
from manim.utils.caching import handle_caching_play
from manim.utils.color import color_to_rgba
from manim.utils.exceptions import EndSceneEarlyException

from ..constants import *
from ..scene.scene_file_writer import SceneFileWriter
from ..utils import opengl
from ..utils.config_ops import _Data
from ..utils.simple_functions import clip
from ..utils.space_ops import (
    angle_of_vector,
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

__all__ = ["OpenGLCamera", "OpenGLRenderer"]


class OpenGLCamera(OpenGLMobject):
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


class OpenGLRenderer:
    def __init__(
        self,
        file_writer_class: type[SceneFileWriter] = SceneFileWriter,
        skip_animations: bool = False,
    ) -> None:
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
                with contextlib.suppress(KeyError):
                    shader.set_uniform(name, value)
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

    def update_skipping_status(self) -> None:
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
    def play(self, scene, *args, **kwargs):
        # TODO: Handle data locking / unlocking.
        self.animation_start_time = time.time()
        self.file_writer.begin_animation(not self.skip_animations)

        scene.compile_animation_data(*args, **kwargs)
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
            if not mobject.should_render:
                continue
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
        fh = config["frame_height"]
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
