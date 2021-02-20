import moderngl
from .opengl_renderer_window import Window
from .shader_wrapper import ShaderWrapper
import numpy as np
from ..mobject.types.vectorized_mobject import VMobject
import itertools as it
import time

from ..mobject import opengl_geometry


class OpenGLCamera:
    use_z_index = True
    frame_rate = 60


points_per_curve = 3
JOINT_TYPE_MAP = {
    "auto": 0,
    "round": 1,
    "bevel": 2,
    "miter": 3,
}


class OpenGLRenderer:
    def __init__(self):
        self.num_plays = 0
        self.skip_animations = False

        self.window = Window(size=(854, 480))

        self.camera = OpenGLCamera()

        self.context = self.window.ctx
        self.context.enable(moderngl.BLEND)
        self.context.blend_func = (
            moderngl.SRC_ALPHA,
            moderngl.ONE_MINUS_SRC_ALPHA,
            moderngl.ONE,
            moderngl.ONE,
        )
        self.frame_buffer_object = self.context.detect_framebuffer()
        self.id_to_shader_program = {}

    def update_depth_test(self, context, shader_wrapper):
        if shader_wrapper.depth_test:
            self.context.enable(moderngl.DEPTH_TEST)
        else:
            self.context.disable(moderngl.DEPTH_TEST)

    def render_mobject(self, mob):
        shader_wrapper_list = self.get_shader_wrapper_list(mob)
        render_group_list = map(
            lambda x: self.get_render_group(self.context, x), shader_wrapper_list
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

    def read_data_to_shader(self, mob, shader_data, shader_data_key, data_key):
        # Check data alignment.
        d_len = len(mob.data[data_key])
        if d_len != 1 and d_len != len(shader_data):
            mob.data[data_key] = resize_with_interpolation(
                mob.data[data_key], len(shader_data)
            )

        shader_data[shader_data_key] = mob.data[data_key]

    def get_stroke_shader_data(self, mob):
        points = mob.data["points"]
        stroke_data = np.zeros(len(points), dtype=VMobject.stroke_dtype)

        stroke_data["point"] = points
        stroke_data["prev_point"][:points_per_curve] = points[-points_per_curve:]
        stroke_data["prev_point"][points_per_curve:] = points[:-points_per_curve]
        stroke_data["next_point"][:-points_per_curve] = points[points_per_curve:]
        stroke_data["next_point"][-points_per_curve:] = points[:points_per_curve]

        self.read_data_to_shader(mob, stroke_data, "color", "stroke_rgba")
        self.read_data_to_shader(mob, stroke_data, "stroke_width", "stroke_width")
        self.read_data_to_shader(mob, stroke_data, "unit_normal", "unit_normal")

        return stroke_data

    def get_fill_shader_data(self, mob):
        points = mob.data["points"]
        fill_data = np.zeros(len(points), dtype=VMobject.fill_dtype)
        fill_data["vert_index"][:, 0] = range(len(points))

        self.read_data_to_shader(mob, fill_data, "point", "points")
        self.read_data_to_shader(mob, fill_data, "color", "fill_rgba")
        self.read_data_to_shader(mob, fill_data, "unit_normal", "unit_normal")

        return fill_data

    def get_stroke_shader_wrapper(self, mob):
        return ShaderWrapper(
            vert_data=self.get_stroke_shader_data(mob),
            shader_folder="quadratic_bezier_stroke",
            render_primitive=moderngl.TRIANGLES,
            uniforms=self.get_stroke_uniforms(mob),
            depth_test=mob.depth_test,
        )

    def get_fill_shader_wrapper(self, mob):
        return ShaderWrapper(
            vert_data=self.get_fill_shader_data(mob),
            vert_indices=self.get_triangulation(mob),
            shader_folder="quadratic_bezier_fill",
            render_primitive=moderngl.TRIANGLES,
            uniforms=self.get_fill_uniforms(mob),
            depth_test=mob.depth_test,
        )

    def get_shader_wrapper_list(self, mob):
        fill_shader_wrappers = []
        stroke_shader_wrappers = []
        back_stroke_shader_wrappers = []
        if any(mob.data["fill_rgba"][:, 3]):
            fill_shader_wrappers.append(self.get_fill_shader_wrapper(mob))
        if any(mob.data["stroke_rgba"][:, 3]) and mob.data["stroke_width"][0]:
            stroke_shader_wrapper = self.get_stroke_shader_wrapper(mob)
            # TODO: Handle background_stroke
            # if mob.draw_stroke_behind_fill:
            #     back_stroke_shader_wrappers.append(ssw)
            # else:
            stroke_shader_wrappers.append(stroke_shader_wrapper)

        # Combine data lists
        wrapper_lists = [
            back_stroke_shader_wrappers,
            fill_shader_wrappers,
            stroke_shader_wrappers,
        ]

        result = []
        for wrapper_list in wrapper_lists:
            if wrapper_list:
                wrapper = wrapper_list[0]
                wrapper.combine_with(*wrapper_list[1:])
                result.append(wrapper)
        return result

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

    def get_triangulation(self, mob, normal_vector=None):
        # Figure out how to triangulate the interior to know
        # how to send the points as to the vertex shader.
        # First triangles come directly from the points
        if normal_vector is None:
            normal_vector = mob.get_unit_normal()

        if not mob.needs_new_triangulation:
            return mob.triangulation

        points = mob.data["points"]

        if len(points) <= 1:
            mob.triangulation = np.zeros(0, dtype="i4")
            mob.needs_new_triangulation = False
            return mob.triangulation

        if not np.isclose(normal_vector, OUT).all():
            # Rotate points such that unit normal vector is OUT
            points = np.dot(points, z_to_vector(normal_vector))
        indices = np.arange(len(points), dtype=int)

        b0s = points[0::3]
        b1s = points[1::3]
        b2s = points[2::3]
        v01s = b1s - b0s
        v12s = b2s - b1s

        crosses = cross2d(v01s, v12s)
        convexities = np.sign(crosses)

        atol = tolerance_for_point_equality
        end_of_loop = np.zeros(len(b0s), dtype=bool)
        end_of_loop[:-1] = (np.abs(b2s[:-1] - b0s[1:]) > atol).any(1)
        end_of_loop[-1] = True

        concave_parts = convexities < 0

        # These are the vertices to which we'll apply a polygon triangulation
        inner_vert_indices = np.hstack(
            [
                indices[0::3],
                indices[1::3][concave_parts],
                indices[2::3][end_of_loop],
            ]
        )
        inner_vert_indices.sort()
        rings = np.arange(1, len(inner_vert_indices) + 1)[inner_vert_indices % 3 == 2]

        # Triangulate
        inner_verts = points[inner_vert_indices]
        inner_tri_indices = inner_vert_indices[
            earclip_triangulation(inner_verts, rings)
        ]

        tri_indices = np.hstack([indices, inner_tri_indices])
        mob.triangulation = tri_indices
        mob.needs_new_triangulation = False
        return tri_indices

    def get_stroke_uniforms(self, mob):
        return dict(
            **self.get_fill_uniforms(mob),
            joint_type=JOINT_TYPE_MAP[mob.joint_type],
            flat_stroke=float(mob.flat_stroke),
        )

    def get_fill_uniforms(self, mob):
        return dict(
            is_fixed_in_frame=float(mob.is_fixed_in_frame),
            gloss=mob.gloss,
            shadow=mob.shadow,
        )

    def set_shader_uniforms(self, shader, shader_wrapper):
        perspective_uniforms = {
            "frame_shape": (14.222222222222221, 8.0),
            "anti_alias_width": 0.016666666666666666,
            "camera_center": (0.0, 0.0, 0.0),
            "camera_rotation": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            "light_source_position": (-10.0, 10.0, 10.0),
            "focal_distance": 16.0,
        }

        # for name, path in shader_wrapper.texture_paths.items():
        #     tid = self.get_texture_id(path)
        #     shader[name].value = tid
        for name, value in it.chain(
            shader_wrapper.uniforms.items(), perspective_uniforms.items()
        ):
            try:
                shader[name].value = value
            except KeyError:
                pass

    def init_scene(self, scene):
        self.file_writer = None

    def play(self, scene, *args, **kwargs):
        if scene.compile_animation_data(*args, **kwargs):
            self.animation_start_time = time.time()
            self.animation_elapsed_time = 0
            scene.play_internal()
        self.num_plays += 1

    def render(self, scene, frame_offset, moving_mobjects):
        def update_frame():
            self.frame_buffer_object.clear(*window_background_color)
            self.render_mobject(scene.mobjects[0])
            self.window.swap_buffers()
            self.animation_elapsed_time = time.time() - self.animation_start_time

        window_background_color = (0.2, 0.2, 0.2, 1)
        update_frame()
        while self.animation_elapsed_time < frame_offset:
            update_frame()

    def scene_finished(self, scene):
        pass

    def save_static_frame_data(self, scene, static_mobjects):
        pass
