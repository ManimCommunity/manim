import itertools as it
import time

import moderngl
import numpy as np
from PIL import Image

from manim import config
from manim.renderer.cairo_renderer import handle_play_like_call
from manim.utils.caching import handle_caching_play
from manim.utils.color import color_to_rgba
from manim.utils.exceptions import EndSceneEarlyException

from ..constants import *
from ..mobject.opengl_mobject import OpenGLMobject, OpenGLPoint
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


class OpenGLCamera(OpenGLMobject):
    def __init__(
        self,
        frame_shape=None,
        center_point=None,
        # Theta, phi, gamma
        euler_angles=[0, 0, 0],
        focal_distance=2,
        light_source_position=[-10, 10, 10],
        **kwargs,
    ):
        self.use_z_index = True
        self.frame_rate = 60

        if frame_shape is None:
            self.frame_shape = (config["frame_width"], config["frame_height"])
        else:
            self.frame_shape = frame_shape

        if center_point is None:
            self.center_point = ORIGIN
        else:
            self.center_point = center_point

        if euler_angles is None:
            self.euler_angles = [0, 0, 0]
        else:
            self.euler_angles = euler_angles

        self.focal_distance = focal_distance

        if light_source_position is None:
            self.light_source_position = [-10, 10, 10]
        else:
            self.light_source_position = light_source_position
        self.light_source = OpenGLPoint(self.light_source_position)

        self.model_matrix = opengl.translation_matrix(0, 0, 11)

        super().__init__(**kwargs)

    def get_position(self):
        return self.model_matrix[:, 3][:3]

    def get_view_matrix(self, format=True):
        if format:
            return opengl.matrix_to_shader_input(np.linalg.inv(self.model_matrix))
        else:
            return np.linalg.inv(self.model_matrix)

    def init_data(self):
        super().init_data()
        self.data["euler_angles"] = np.array(self.euler_angles, dtype=float)
        self.refresh_rotation_matrix()

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
        self.model_matrix = opengl.translation_matrix(0, 0, 11)
        return self

    def refresh_rotation_matrix(self):
        # Rotate based on camera orientation
        theta, phi, gamma = self.data["euler_angles"]
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
            self.data["euler_angles"][0] = theta
        if phi is not None:
            self.data["euler_angles"][1] = phi
        if gamma is not None:
            self.data["euler_angles"][2] = gamma
        self.refresh_rotation_matrix()
        return self

    def set_theta(self, theta):
        return self.set_euler_angles(theta=theta)

    def set_phi(self, phi):
        return self.set_euler_angles(phi=phi)

    def set_gamma(self, gamma):
        return self.set_euler_angles(gamma=gamma)

    def increment_theta(self, dtheta):
        self.data["euler_angles"][0] += dtheta
        self.refresh_rotation_matrix()
        return self

    def increment_phi(self, dphi):
        phi = self.data["euler_angles"][1]
        new_phi = clip(phi + dphi, -PI / 2, PI / 2)
        self.data["euler_angles"][1] = new_phi
        self.refresh_rotation_matrix()
        return self

    def increment_gamma(self, dgamma):
        self.data["euler_angles"][2] += dgamma
        self.refresh_rotation_matrix()
        return self

    def get_shape(self):
        return (self.get_width(), self.get_height())

    def get_center(self):
        # Assumes first point is at the center
        return self.get_points()[0]

    def get_width(self):
        points = self.get_points()
        return points[2, 0] - points[1, 0]

    def get_height(self):
        points = self.get_points()
        return points[4, 1] - points[3, 1]

    def get_focal_distance(self):
        return self.focal_distance * self.get_height()

    def interpolate(self, *args, **kwargs):
        super().interpolate(*args, **kwargs)
        self.refresh_rotation_matrix()


points_per_curve = 3
JOINT_TYPE_MAP = {
    "auto": 0,
    "round": 1,
    "bevel": 2,
    "miter": 3,
}


class OpenGLRenderer:
    def __init__(self, skip_animations=False):
        # Measured in pixel widths, used for vector graphics
        self.anti_alias_width = 1.5

        self._original_skipping_status = skip_animations
        self.skip_animations = skip_animations
        self.animations_hashes = []
        self.num_plays = 0

        self.camera = OpenGLCamera()
        self.pressed_keys = set()

        # Initialize texture map.
        self.path_to_texture_id = {}

    def init_scene(self, scene):
        self.partial_movie_files = []
        self.file_writer = SceneFileWriter(
            self,
            scene.__class__.__name__,
        )
        self.scene = scene
        if not hasattr(self, "window"):
            if config["preview"]:
                self.window = Window(self)
                self.context = self.window.ctx
                self.frame_buffer_object = self.context.detect_framebuffer()
            else:
                self.window = None
                self.context = moderngl.create_standalone_context()
                self.frame_buffer_object = self.get_frame_buffer_object(self.context, 0)
                self.frame_buffer_object.use()
            self.context.enable(moderngl.BLEND)
            self.context.blend_func = (
                moderngl.SRC_ALPHA,
                moderngl.ONE_MINUS_SRC_ALPHA,
                moderngl.ONE,
                moderngl.ONE,
            )

    def get_pixel_shape(self):
        return self.frame_buffer_object.viewport[2:4]

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
                shader_wrapper.uniforms.items(), self.perspective_uniforms.items()
            ):
                try:
                    shader.set_uniform(name, value)
                except KeyError:
                    pass
            try:
                shader.set_uniform("u_view_matrix", self.scene.camera.get_view_matrix())
                shader.set_uniform(
                    "u_projection_matrix", opengl.orthographic_projection_matrix()
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
            )
            mesh.render()

    def get_texture_id(self, path):
        if path not in self.path_to_texture_id:
            # A way to increase tid's sequentially
            tid = len(self.path_to_texture_id)
            im = Image.open(path)
            texture = self.context.texture(
                size=im.size,
                components=len(im.getbands()),
                data=im.tobytes(),
            )
            texture.use(location=tid)
            self.path_to_texture_id[path] = tid
        return self.path_to_texture_id[path]

    def update_skipping_status(self):
        """
        This method is used internally to check if the current
        animation needs to be skipped or not. It also checks if
        the number of animations that were played correspond to
        the number of animations that need to be played, and
        raises an EndSceneEarlyException if they don't correspond.
        """
        if config["from_animation_number"]:
            if self.num_plays < config["from_animation_number"]:
                self.skip_animations = True
        if config["upto_animation_number"]:
            if self.num_plays > config["upto_animation_number"]:
                self.skip_animations = True
                raise EndSceneEarlyException()

    @handle_caching_play
    @handle_play_like_call
    def play(self, scene, *args, **kwargs):
        # TODO: Handle data locking / unlocking.
        if scene.compile_animation_data(*args, **kwargs):
            scene.begin_animations()
            scene.play_internal()

    def clear_screen(self):
        window_background_color = color_to_rgba(config["background_color"])
        self.frame_buffer_object.clear(*window_background_color)
        self.window.swap_buffers()

    def render(self, scene, frame_offset, moving_mobjects):
        def update_frame():
            self.frame_buffer_object.clear(*window_background_color)
            self.refresh_perspective_uniforms(scene.camera)

            for mobject in scene.mobjects:
                self.render_mobject(mobject)

            view_matrix = scene.camera.get_view_matrix()
            for mesh in scene.meshes:
                try:
                    mesh.shader.set_uniform("u_view_matrix", view_matrix)
                    mesh.shader.set_uniform(
                        "u_projection_matrix", opengl.perspective_projection_matrix()
                    )
                except KeyError:
                    pass
                mesh.render()

            self.animation_elapsed_time = time.time() - self.animation_start_time

        window_background_color = color_to_rgba(config["background_color"])
        update_frame()

        if self.skip_animations:
            return

        if config["write_to_movie"]:
            self.file_writer.write_frame(self)

        if self.window is not None:
            self.window.swap_buffers()
            while self.animation_elapsed_time < frame_offset:
                update_frame()
                self.window.swap_buffers()

    def scene_finished(self, scene):
        self.file_writer.finish()

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
                (pixel_width, pixel_height), samples=samples
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
        result_dimensions = (config["pixel_height"], config["pixel_width"], 4)
        np_buf = np.frombuffer(raw, dtype="uint8").reshape(result_dimensions)
        np_buf = np.flipud(np_buf)
        return np_buf

    # Returns offset from the bottom left corner in pixels.
    def pixel_coords_to_space_coords(self, px, py, relative=False):
        pw, ph = config["pixel_width"], config["pixel_height"]
        fw, fh = config["frame_width"], config["frame_height"]
        fc = self.camera.get_center()
        if relative:
            return 2 * np.array([px / pw, py / ph, 0])
        else:
            # Only scale wrt one axis
            scale = fh / ph
            return fc + scale * np.array([(px - pw / 2), (py - ph / 2), 0])
