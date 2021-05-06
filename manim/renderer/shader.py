import os
import re
import textwrap
from pathlib import Path

import moderngl
import numpy as np

from .. import config
from ..utils import opengl
from ..utils.simple_functions import get_parameters

SHADER_FOLDER = Path(__file__).parent / "shaders"
shader_program_cache = {}
file_path_to_code_map = {}


def get_shader_code_from_file(file_path):
    if file_path in file_path_to_code_map:
        return file_path_to_code_map[file_path]
    with open(file_path, "r") as f:
        source = f.read()
        include_lines = re.finditer(
            r"^#include (?P<include_path>.*\.glsl)$", source, flags=re.MULTILINE
        )
        for match in include_lines:
            include_path = match.group("include_path")
            included_code = get_shader_code_from_file(
                os.path.join(file_path.parent / include_path)
            )
            source = source.replace(match.group(0), included_code)
        file_path_to_code_map[file_path] = source
        return source


class Mesh:
    def __init__(self, shader, attributes, indices=None, use_depth_test=True):
        self.shader = shader
        self.attributes = attributes
        self.indices = indices
        self.use_depth_test = use_depth_test
        self.init_updaters()

        self.translation = np.zeros(3)
        self.rotation = np.zeros(3)
        self.scale = np.zeros(3)
        self.model_matrix = np.eye(4)
        self.model_matrix_needs_update = False

    def render(self):
        # Set matrix uniforms.
        if self.model_matrix_needs_update:
            pass
        try:
            self.shader.set_uniform(
                "u_model_matrix", opengl.matrix_to_shader_input(self.model_matrix)
            )
        except KeyError:
            pass

        if self.use_depth_test:
            self.shader.context.enable(moderngl.DEPTH_TEST)

        vertex_buffer_object = self.shader.context.buffer(self.attributes.tobytes())
        if self.indices is None:
            index_buffer_object = None
        else:
            vert_index_data = self.indices.astype("i4").tobytes()
            if vert_index_data:
                index_buffer_object = self.shader.context.buffer(vert_index_data)
            else:
                index_buffer_object = None
        vertex_array_object = self.shader.context.simple_vertex_array(
            self.shader.shader_program,
            vertex_buffer_object,
            *self.attributes.dtype.names,
            index_buffer=index_buffer_object,
        )
        vertex_array_object.render(moderngl.TRIANGLES)
        vertex_buffer_object.release()
        vertex_array_object.release()
        if index_buffer_object is not None:
            index_buffer_object.release()

    def init_updaters(self):
        self.time_based_updaters = []
        self.non_time_updaters = []
        self.has_updaters = False
        self.updating_suspended = False

    def update(self, dt=0):
        if not self.has_updaters or self.updating_suspended:
            return self
        for updater in self.time_based_updaters:
            updater(self, dt)
        for updater in self.non_time_updaters:
            updater(self)
        return self

    def get_time_based_updaters(self):
        return self.time_based_updaters

    def has_time_based_updater(self):
        return len(self.time_based_updaters) > 0

    def get_updaters(self):
        return self.time_based_updaters + self.non_time_updaters

    def add_updater(self, update_function, index=None, call_updater=True):
        if "dt" in get_parameters(update_function):
            updater_list = self.time_based_updaters
        else:
            updater_list = self.non_time_updaters

        if index is None:
            updater_list.append(update_function)
        else:
            updater_list.insert(index, update_function)

        self.refresh_has_updater_status()
        if call_updater:
            self.update()
        return self

    def remove_updater(self, update_function):
        for updater_list in [self.time_based_updaters, self.non_time_updaters]:
            while update_function in updater_list:
                updater_list.remove(update_function)
        self.refresh_has_updater_status()
        return self

    def clear_updaters(self):
        self.time_based_updaters = []
        self.non_time_updaters = []
        self.refresh_has_updater_status()
        return self

    def match_updaters(self, mobject):
        self.clear_updaters()
        for updater in mobject.get_updaters():
            self.add_updater(updater)
        return self

    def suspend_updating(self):
        self.updating_suspended = True
        return self

    def resume_updating(self, call_updater=True):
        self.updating_suspended = False
        if call_updater:
            self.update(dt=0)
        return self

    def refresh_has_updater_status(self):
        self.has_updaters = len(self.get_updaters()) > 0
        return self


class Shader:
    def __init__(
        self,
        context,
        name=None,
        source=None,
    ):
        self.context = context
        self.name = name

        # See if the program is cached.
        if self.name in shader_program_cache:
            self.shader_program = shader_program_cache[self.name]

        # Generate the shader from inline code if it was passed.
        if source is not None:
            self.shader_program = context.program(**source)
        else:
            # Search for a file containing the shader.
            source_dict = {}
            source_dict_key = {
                "vert": "vertex_shader",
                "frag": "fragment_shader",
                "geom": "geometry_shader",
            }
            shader_folder = SHADER_FOLDER / name
            for shader_file in shader_folder.iterdir():
                shader_file_path = shader_folder / shader_file
                shader_source = get_shader_code_from_file(shader_file_path)
                source_dict[source_dict_key[shader_file_path.stem]] = shader_source
            self.shader_program = context.program(**source_dict)

        # Cache the shader.
        if name is not None:
            shader_program_cache[self.name] = self.shader_program

    def set_uniform(self, name, value):
        self.shader_program[name] = value


class FullScreenQuad(Mesh):
    def __init__(
        self,
        context,
        fragment_shader_source=None,
        fragment_shader_name=None,
        output_color_variable="frag_color",
    ):
        if fragment_shader_source is None and fragment_shader_name is None:
            raise Exception("Must either pass shader name or shader source.")

        if fragment_shader_name is not None:
            # Use the name.
            shader_file_path = SHADER_FOLDER / f"{fragment_shader_name}.frag"
            fragment_shader_source = get_shader_code_from_file(shader_file_path)
        elif fragment_shader_source is not None:
            fragment_shader_source = textwrap.dedent(fragment_shader_source.lstrip())
            fragment_shader_lines = fragment_shader_source.split("\n")

            # If the first line is a version string, insert after it.
            insertion_index = 0
            if fragment_shader_lines[0].startswith("#version"):
                insertion_index += 1
            fragment_shader_lines.insert(
                insertion_index,
                f"out vec4 {output_color_variable};",
            )
            fragment_shader_source = "\n".join(fragment_shader_lines)

        shader = Shader(
            context,
            source=dict(
                vertex_shader="""
                #version 330

                in vec4 in_vert;
                uniform mat4 u_model_view_matrix;
                uniform mat4 u_projection_matrix;

                void main() {{
                    vec4 camera_space_vertex = u_model_view_matrix * in_vert;
                    vec4 clip_space_vertex = u_projection_matrix * camera_space_vertex;
                    gl_Position = clip_space_vertex;
                }}
                """,
                fragment_shader=fragment_shader_source,
            ),
        )
        shader.set_uniform("u_model_view_matrix", opengl.view_matrix())
        shader.set_uniform(
            "u_projection_matrix", opengl.orthographic_projection_matrix()
        )
        super().__init__(shader, None)

    def render(self):
        self.attributes = np.zeros(6, dtype=[("in_vert", np.float32, (4,))])
        self.attributes["in_vert"] = np.array(
            [
                [-config["frame_x_radius"], -config["frame_y_radius"], 0, 1],
                [-config["frame_x_radius"], config["frame_y_radius"], 0, 1],
                [config["frame_x_radius"], config["frame_y_radius"], 0, 1],
                [-config["frame_x_radius"], -config["frame_y_radius"], 0, 1],
                [config["frame_x_radius"], -config["frame_y_radius"], 0, 1],
                [config["frame_x_radius"], config["frame_y_radius"], 0, 1],
            ],
        )
        super().render()
