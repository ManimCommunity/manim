import itertools as it
import time

import moderngl
import numpy as np
from PIL import Image

from manim import config, logger
from manim.utils.color import color_to_rgba
from manim.utils.exceptions import EndSceneEarlyException, RerunSceneException

from .renderer import Renderer
from ..constants import *
from ..mobject.opengl_mobject import OpenGLMobject, OpenGLPoint
from ..mobject.types.opengl_vectorized_mobject import OpenGLVMobject
from ..utils import opengl, space_ops
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

    def get_view_matrix(self, format=True):
        if format:
            return opengl.matrix_to_shader_input(np.linalg.inv(self.model_matrix))
        else:
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
JOINT_TYPE_MAP = {
    "auto": 0,
    "round": 1,
    "bevel": 2,
    "miter": 3,
}


class OpenGLRenderer(Renderer):

    def __init__(self, skip_animations=False, widgets=None):
        # Measured in pixel widths, used for vector graphics
        if widgets is None:
            widgets = []
        self.anti_alias_width = 1.5

        self._original_skipping_status = skip_animations
        self.skip_animations = skip_animations
        self.animation_start_time = 0
        self.animation_elapsed_time = 0
        self.time = 0
        self.animations_hashes = []
        self.num_plays = 0
        self.interactive_mode = False
        self.widgets = [] if not widgets else widgets
        self.mouse_press_callbacks = []
        self.pressed_keys = set()
        self.key_to_function_map = {}

        self.camera_target = ORIGIN
        self.camera = OpenGLCamera()

        # Initialize texture map.
        self.path_to_texture_id = {}

        self._background_color = color_to_rgba(config["background_color"], 1.0)

    def init_scene(self, scene=None):
        if not hasattr(self, "window"):
            if self.should_create_window():
                from .opengl_renderer_window import Window

                self.mouse_point = OpenGLPoint()
                self.mouse_drag_point = OpenGLPoint()
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
                shader.set_uniform("u_view_matrix", self.camera.get_view_matrix())
                shader.set_uniform(
                    "u_projection_matrix",
                    self.camera.projection_matrix,
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

    def clear_screen(self):
        self.frame_buffer_object.clear(*self.background_color)
        self.window.swap_buffers()

    def before_animation(self):
        self.animation_start_time = time.time()

    def has_interaction(self):
        return True

    def can_handle_static_wait(self):
        return False

    def after_animation(self):
        pass

    def before_render(self):
        pass

    def after_render(self):
        pass

    def render(
        self,
        frame_offset,
        moving_mobjects,
        skip_animations=False,
        mobjects=None,
        meshes=None,
        file_writer=None,
        foreground_mobjects=None,
    ):
        self.update_frame(moving_mobjects, meshes=meshes, mobjects=mobjects)
        if skip_animations:
            return

        file_writer.write_frame(self)

        if self.window is not None:
            self.window.swap_buffers()
            while self.animation_elapsed_time < frame_offset:
                self.update_frame(moving_mobjects, meshes=meshes, mobjects=mobjects)
                self.window.swap_buffers()

    def update_frame(self,
        moving_mobjects,
        skip_animations=False,
        include_submobjects=True,
        ignore_skipping=False,
        mobjects=None,
        meshes=None,
        file_writer=None,
        foreground_mobjects=None,
        **kwargs):
        self.frame_buffer_object.clear(*self.background_color)
        self.refresh_perspective_uniforms(self.camera)

        for mobject in mobjects:
            if not mobject.should_render:
                continue
            self.render_mobject(mobject)

        for obj in meshes:
            for mesh in obj.get_meshes():
                mesh.set_uniforms(self)
                mesh.render()

        self.animation_elapsed_time = time.time() - self.animation_start_time

    def should_save_last_frame(self, num_plays):
        if config["save_last_frame"]:
            return True
        if self.interactive_mode:
            return False
        return num_plays == 0

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

    def save_static_frame_data(
        self, static_mobjects, mobjects=None, foreground_mobjects=None
    ):
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

    def freeze_current_frame(self, duration: float, file_writer, skip_animations=False):
        pass

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
    def pixel_coords_to_space_coords(self, px, py, relative=False):
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
            return fc + scale * np.array([(px - pw / 2), (py - ph / 2), 0])

    @property
    def background_color(self):
        return self._background_color

    @background_color.setter
    def background_color(self, value):
        self._background_color = color_to_rgba(value, 1.0)

    def on_mouse_motion(self, point, d_point):
        self.mouse_point.move_to(point)
        if SHIFT_VALUE in self.pressed_keys:
            shift = -d_point
            shift[0] *= self.camera.get_width() / 2
            shift[1] *= self.camera.get_height() / 2
            transform = self.camera.inverse_rotation_matrix
            shift = np.dot(np.transpose(transform), shift)
            self.camera.shift(shift)

    def on_mouse_scroll(self, point, offset):
        if not config.use_projection_stroke_shaders:
            factor = 1 + np.arctan(-2.1 * offset[1])
            self.camera.scale(factor, about_point=self.camera_target)
        self.mouse_scroll_orbit_controls(point, offset)

    def on_key_press(self, symbol, modifiers):
        try:
            char = chr(symbol)
        except OverflowError:
            logger.warning("The value of the pressed key is too large.")
            return

        if char == "r":
            self.camera.to_default_state()
            self.camera_target = np.array([0, 0, 0], dtype=np.float32)
        elif char == "q":
            self.quit_interaction = True
        else:
            if char in self.key_to_function_map:
                self.key_to_function_map[char]()

    def on_key_release(self, symbol, modifiers):
        pass

    def on_mouse_drag(self, point, d_point, buttons, modifiers):
        self.mouse_drag_point.move_to(point)
        if buttons == 1:
            self.camera.increment_theta(-d_point[0])
            self.camera.increment_phi(d_point[1])
        elif buttons == 4:
            camera_x_axis = self.camera.model_matrix[:3, 0]
            horizontal_shift_vector = -d_point[0] * camera_x_axis
            vertical_shift_vector = -d_point[1] * np.cross(OUT, camera_x_axis)
            total_shift_vector = horizontal_shift_vector + vertical_shift_vector
            self.camera.shift(1.1 * total_shift_vector)

        self.mouse_drag_orbit_controls(point, d_point, buttons, modifiers)

    def mouse_scroll_orbit_controls(self, point, offset):
        camera_to_target = self.camera_target - self.camera.get_position()
        camera_to_target *= np.sign(offset[1])
        shift_vector = 0.01 * camera_to_target
        self.camera.model_matrix = (
            opengl.translation_matrix(*shift_vector) @ self.camera.model_matrix
        )

    def mouse_drag_orbit_controls(self, point, d_point, buttons, modifiers):
        # Left click drag.
        if buttons == 1:
            # Translate to target the origin and rotate around the z axis.
            self.camera.model_matrix = (
                opengl.rotation_matrix(z=-d_point[0])
                @ opengl.translation_matrix(*-self.camera_target)
                @ self.camera.model_matrix
            )

            # Rotation off of the z axis.
            camera_position = self.camera.get_position()
            camera_y_axis = self.camera.model_matrix[:3, 1]
            axis_of_rotation = space_ops.normalize(
                np.cross(camera_y_axis, camera_position),
            )
            rotation_matrix = space_ops.rotation_matrix(
                d_point[1],
                axis_of_rotation,
                homogeneous=True,
            )

            maximum_polar_angle = self.camera.maximum_polar_angle
            minimum_polar_angle = self.camera.minimum_polar_angle

            potential_camera_model_matrix = rotation_matrix @ self.camera.model_matrix
            potential_camera_location = potential_camera_model_matrix[:3, 3]
            potential_camera_y_axis = potential_camera_model_matrix[:3, 1]
            sign = (
                np.sign(potential_camera_y_axis[2])
                if potential_camera_y_axis[2] != 0
                else 1
            )
            potential_polar_angle = sign * np.arccos(
                potential_camera_location[2]
                / np.linalg.norm(potential_camera_location),
            )
            if minimum_polar_angle <= potential_polar_angle <= maximum_polar_angle:
                self.camera.model_matrix = potential_camera_model_matrix
            else:
                sign = np.sign(camera_y_axis[2]) if camera_y_axis[2] != 0 else 1
                current_polar_angle = sign * np.arccos(
                    camera_position[2] / np.linalg.norm(camera_position),
                )
                if potential_polar_angle > maximum_polar_angle:
                    polar_angle_delta = maximum_polar_angle - current_polar_angle
                else:
                    polar_angle_delta = minimum_polar_angle - current_polar_angle
                rotation_matrix = space_ops.rotation_matrix(
                    polar_angle_delta,
                    axis_of_rotation,
                    homogeneous=True,
                )
                self.camera.model_matrix = rotation_matrix @ self.camera.model_matrix

            # Translate to target the original target.
            self.camera.model_matrix = (
                opengl.translation_matrix(*self.camera_target)
                @ self.camera.model_matrix
            )
        # Right click drag.
        elif buttons == 4:
            camera_x_axis = self.camera.model_matrix[:3, 0]
            horizontal_shift_vector = -d_point[0] * camera_x_axis
            vertical_shift_vector = -d_point[1] * np.cross(OUT, camera_x_axis)
            total_shift_vector = horizontal_shift_vector + vertical_shift_vector

            self.camera.model_matrix = (
                opengl.translation_matrix(*total_shift_vector)
                @ self.camera.model_matrix
            )
            self.camera_target += total_shift_vector

    def set_key_function(self, char, func):
        self.key_to_function_map[char] = func

    def on_mouse_press(self, point, button, modifiers):
        for func in self.mouse_press_callbacks:
            func()