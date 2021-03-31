from manim.utils.exceptions import EndSceneEarlyException
from manim.utils.caching import handle_caching_play
from manim.renderer.cairo_renderer import handle_play_like_call
from manim.utils.color import color_to_rgba
import moderngl
from .opengl_renderer_window import Window
from .shader_wrapper import ShaderWrapper
import numpy as np
from ..mobject.types.vectorized_mobject import VMobject
import itertools as it
import time
from .. import logger
from ..constants import *
from ..utils.space_ops import (
    cross2d,
    earclip_triangulation,
    z_to_vector,
    quaternion_mult,
    quaternion_from_angle_axis,
    rotation_matrix_transpose_from_quaternion,
    rotation_matrix_transpose,
    angle_of_vector,
)
from ..utils.simple_functions import clip

from ..mobject import opengl_geometry
from ..mobject.opengl_mobject import OpenGLMobject, OpenGLPoint
from PIL import Image
from manim import config
from ..scene.scene_file_writer import SceneFileWriter


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

        super().__init__(**kwargs)

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
        new_phi = clip(phi + dphi, 0, PI)
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

        # Initialize shader map.
        self.id_to_shader_program = {}

        # Initialize texture map.
        self.path_to_texture_id = {}

    def init_scene(self, scene):
        self.partial_movie_files = []
        self.file_writer = SceneFileWriter(
            self,
            scene.__class__.__name__,
        )
        self.scene = scene
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

        # Initialize shader map.
        self.id_to_shader_program = {}

        # Initialize texture map.
        self.path_to_texture_id = {}

    def update_depth_test(self, context, shader_wrapper):
        if shader_wrapper.depth_test:
            self.context.enable(moderngl.DEPTH_TEST)
        else:
            self.context.disable(moderngl.DEPTH_TEST)

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

    def render_mobjects(self, mobs):
        for mob in mobs:
            shader_wrapper_list = mob.get_shader_wrapper_list()
            render_group_list = map(
                lambda shader_wrapper: self.get_render_group(
                    self.context, shader_wrapper
                ),
                shader_wrapper_list,
            )
            for render_group in render_group_list:
                self.render_render_group(render_group)

    def render_render_group(self, render_group):
        shader_wrapper = render_group["shader_wrapper"]
        shader_program = render_group["prog"]
        self.set_shader_uniforms(render_group["prog"], render_group["shader_wrapper"])
        self.update_depth_test(self.context, shader_wrapper)
        render_group["vao"].render(int(shader_wrapper.render_primitive))

        if render_group["single_use"]:
            for key in ["vbo", "ibo", "vao"]:
                if render_group[key] is not None:
                    render_group[key].release()

    def get_render_group(self, context, shader_wrapper, single_use=True):
        # Data buffers
        vertex_buffer_object = self.context.buffer(shader_wrapper.vert_data.tobytes())
        if shader_wrapper.vert_indices is None:
            index_buffer_object = None
        else:
            vert_index_data = shader_wrapper.vert_indices.astype("i4").tobytes()
            if vert_index_data:
                index_buffer_object = self.context.buffer(vert_index_data)
            else:
                index_buffer_object = None

        # Program and vertex array
        shader_program, vert_format = self.get_shader_program(
            self.context, shader_wrapper
        )
        vertex_array_object = self.context.vertex_array(
            program=shader_program,
            content=[
                (vertex_buffer_object, vert_format, *shader_wrapper.vert_attributes)
            ],
            index_buffer=index_buffer_object,
        )
        return {
            "vbo": vertex_buffer_object,
            "ibo": index_buffer_object,
            "vao": vertex_array_object,
            "prog": shader_program,
            "shader_wrapper": shader_wrapper,
            "single_use": single_use,
        }

    def get_shader_program(self, context, shader_wrapper):
        sid = shader_wrapper.get_program_id()
        if sid not in self.id_to_shader_program:
            # Create shader program for the first time, then cache
            # in self.id_to_shader_program.
            program_code = shader_wrapper.get_program_code()
            program = self.context.program(**program_code)
            vert_format = moderngl.detect_format(
                program, shader_wrapper.vert_attributes
            )
            self.id_to_shader_program[sid] = (program, vert_format)
        return self.id_to_shader_program[sid]

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

    def set_shader_uniforms(self, shader, shader_wrapper):
        # perspective_uniforms = {
        #     "frame_shape": (14.222222222222221, 8.0),
        #     "anti_alias_width": 0.016666666666666666,
        #     "camera_center": (0.0, 0.0, 0.0),
        #     "camera_rotation": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        #     "light_source_position": (-10.0, 10.0, 10.0),
        #     "focal_distance": 16.0,
        # }

        for name, path in shader_wrapper.texture_paths.items():
            tid = self.get_texture_id(path)
            shader[name].value = tid
        for name, value in it.chain(
            shader_wrapper.uniforms.items(), self.perspective_uniforms.items()
        ):
            try:
                shader[name].value = value
            except KeyError:
                pass

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

    def render(self, scene, frame_offset, moving_mobjects):
        def update_frame():
            self.frame_buffer_object.clear(*window_background_color)
            self.refresh_perspective_uniforms(scene.camera)
            self.render_mobjects(scene.mobjects)
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
