from __future__ import annotations

import contextlib
import re
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import moderngl
import numpy as np

from manim import config
from manim.utils import opengl
from manim.utils.updaters import MeshUpdaterWrapper

if TYPE_CHECKING:
    from typing_extensions import Self

    from manim.utils.updaters import MeshDtUpdater, MeshUpdater

SHADER_FOLDER = Path(__file__).parent / "shaders"
shader_program_cache: dict = {}
file_path_to_code_map: dict = {}

__all__ = [
    "Object3D",
    "Mesh",
    "Shader",
    "FullScreenQuad",
]


def get_shader_code_from_file(file_path: Path) -> str:
    if file_path in file_path_to_code_map:
        return file_path_to_code_map[file_path]
    source = file_path.read_text()
    include_lines = re.finditer(
        r"^#include (?P<include_path>.*\.glsl)$",
        source,
        flags=re.MULTILINE,
    )
    for match in include_lines:
        include_path = match.group("include_path")
        included_code = get_shader_code_from_file(
            file_path.parent / include_path,
        )
        source = source.replace(match.group(0), included_code)
    file_path_to_code_map[file_path] = source
    return source


def filter_attributes(unfiltered_attributes, attributes):
    # Construct attributes for only those needed by the shader.
    filtered_attributes_dtype = []
    for i, dtype_name in enumerate(unfiltered_attributes.dtype.names):
        if dtype_name in attributes:
            filtered_attributes_dtype.append(
                (
                    dtype_name,
                    unfiltered_attributes.dtype[i].subdtype[0].str,
                    unfiltered_attributes.dtype[i].shape,
                ),
            )

    filtered_attributes = np.zeros(
        unfiltered_attributes[unfiltered_attributes.dtype.names[0]].shape[0],
        dtype=filtered_attributes_dtype,
    )

    for dtype_name in unfiltered_attributes.dtype.names:
        if dtype_name in attributes:
            filtered_attributes[dtype_name] = unfiltered_attributes[dtype_name]

    return filtered_attributes


class Object3D:
    def __init__(self, *children):
        self.model_matrix = np.eye(4)
        self.normal_matrix = np.eye(4)
        self.children = []
        self.parent = None
        self.add(*children)
        self.init_updaters()

    # TODO: Use path_func.
    def interpolate(self, start, end, alpha, _):
        self.model_matrix = (1 - alpha) * start.model_matrix + alpha * end.model_matrix
        self.normal_matrix = (
            1 - alpha
        ) * start.normal_matrix + alpha * end.normal_matrix

    def single_copy(self):
        copy = Object3D()
        copy.model_matrix = self.model_matrix.copy()
        copy.normal_matrix = self.normal_matrix.copy()
        return copy

    def copy(self):
        node_to_copy = {}

        bfs = [self]
        while bfs:
            node = bfs.pop(0)
            bfs.extend(node.children)

            node_copy = node.single_copy()
            node_to_copy[node] = node_copy

            # Add the copy to the copy of the parent.
            if node.parent is not None and node is not self:
                node_to_copy[node.parent].add(node_copy)
        return node_to_copy[self]

    def add(self, *children):
        for child in children:
            if child.parent is not None:
                raise Exception(
                    "Attempt to add child that's already added to another Object3D",
                )
        self.remove(*children, current_children_only=False)
        self.children.extend(children)
        for child in children:
            child.parent = self

    def remove(self, *children, current_children_only=True):
        if current_children_only:
            for child in children:
                if child.parent != self:
                    raise Exception(
                        "Attempt to remove child that isn't added to this Object3D",
                    )
        self.children = list(filter(lambda child: child not in children, self.children))
        for child in children:
            child.parent = None

    def get_position(self):
        return self.model_matrix[:, 3][:3]

    def set_position(self, position):
        self.model_matrix[:, 3][:3] = position
        return self

    def get_meshes(self):
        dfs = [self]
        while dfs:
            parent = dfs.pop()
            if isinstance(parent, Mesh):
                yield parent
            dfs.extend(parent.children)

    def get_family(self):
        dfs = [self]
        while dfs:
            parent = dfs.pop()
            yield parent
            dfs.extend(parent.children)

    def align_data_and_family(self, _):
        pass

    def hierarchical_model_matrix(self):
        if self.parent is None:
            return self.model_matrix

        model_matrices = [self.model_matrix]
        current_object = self
        while current_object.parent is not None:
            model_matrices.append(current_object.parent.model_matrix)
            current_object = current_object.parent
        return np.linalg.multi_dot(list(reversed(model_matrices)))

    def hierarchical_normal_matrix(self):
        if self.parent is None:
            return self.normal_matrix[:3, :3]

        normal_matrices = [self.normal_matrix]
        current_object = self
        while current_object.parent is not None:
            normal_matrices.append(current_object.parent.model_matrix)
            current_object = current_object.parent
        return np.linalg.multi_dot(list(reversed(normal_matrices)))[:3, :3]

    # Updating

    @property
    def updaters(self) -> Sequence[MeshUpdater]:
        return self.get_updaters()

    def init_updaters(self) -> None:
        self.updater_wrappers: Sequence[MeshUpdaterWrapper] = []
        self.has_updaters = False
        self.updating_suspended = False

    def update(self, dt: float = 0) -> Self:
        if not self.has_updaters or self.updating_suspended:
            return self
        for wrapper in self.updater_wrappers:
            if wrapper.is_time_based:
                wrapper.updater(self, dt)
            else:
                wrapper.updater(self)
        return self

    def get_time_based_updaters(self) -> Sequence[MeshDtUpdater]:
        return [
            wrapper.updater
            for wrapper in self.updater_wrappers
            if wrapper.is_time_based
        ]

    def has_time_based_updater(self) -> bool:
        return any(wrapper.is_time_based for wrapper in self.updater_wrappers)

    def get_updaters(self) -> Sequence[MeshUpdater]:
        return [wrapper.updater for wrapper in self.updater_wrappers]

    def add_updater(
        self,
        update_function: MeshUpdater,
        index: int | None = None,
        call_updater: bool = False,
    ) -> Self:
        wrapper = MeshUpdaterWrapper(update_function)
        if index is None:
            self.updater_wrappers.append(wrapper)
        else:
            self.updater_wrappers.insert(index, wrapper)

        self.refresh_has_updater_status()
        if call_updater:
            self.update()
        return self

    def remove_updater(self, update_function: MeshUpdater) -> Self:
        self.updater_wrappers = [
            wrapper
            for wrapper in self.updater_wrappers
            if wrapper.updater != update_function
        ]
        self.refresh_has_updater_status()
        return self

    def clear_updaters(self, recurse: bool = True) -> Self:
        self.updater_wrappers = []
        self.refresh_has_updater_status()
        if recurse:
            for submob in self.submobjects:
                submob.clear_updaters()
        return self

    def match_updaters(self, obj: Object3D) -> Self:
        self.clear_updaters()
        self.updater_wrappers = obj.updater_wrappers.copy()
        return self

    def suspend_updating(self) -> Self:
        self.updating_suspended = True
        return self

    def resume_updating(self, call_updater: bool = True) -> Self:
        self.updating_suspended = False
        if call_updater:
            self.update(dt=0)
        return self

    def refresh_has_updater_status(self) -> Self:
        self.has_updaters = len(self.get_updaters()) > 0
        return self


class Mesh(Object3D):
    def __init__(
        self,
        shader=None,
        attributes=None,
        geometry=None,
        material=None,
        indices=None,
        use_depth_test=True,
        primitive=moderngl.TRIANGLES,
    ):
        super().__init__()
        if shader is not None and attributes is not None:
            self.shader = shader
            self.attributes = attributes
            self.indices = indices
        elif geometry is not None and material is not None:
            self.shader = material
            self.attributes = geometry.attributes
            self.indices = geometry.index
        else:
            raise Exception(
                "Mesh requires either attributes and a Shader or a Geometry and a "
                "Material",
            )
        self.use_depth_test = use_depth_test
        self.primitive = primitive
        self.skip_render = False
        self.init_updaters()

    def single_copy(self):
        copy = Mesh(
            attributes=self.attributes.copy(),
            shader=self.shader,
            indices=self.indices.copy() if self.indices is not None else None,
            use_depth_test=self.use_depth_test,
            primitive=self.primitive,
        )
        copy.skip_render = self.skip_render
        copy.model_matrix = self.model_matrix.copy()
        copy.normal_matrix = self.normal_matrix.copy()
        # TODO: Copy updaters?
        return copy

    def set_uniforms(self, renderer):
        self.shader.set_uniform(
            "u_model_matrix",
            opengl.matrix_to_shader_input(self.model_matrix),
        )
        self.shader.set_uniform("u_view_matrix", renderer.camera.formatted_view_matrix)
        self.shader.set_uniform(
            "u_projection_matrix",
            renderer.camera.projection_matrix,
        )

    def render(self):
        if self.skip_render:
            return

        if self.use_depth_test:
            self.shader.context.enable(moderngl.DEPTH_TEST)
        else:
            self.shader.context.disable(moderngl.DEPTH_TEST)

        from moderngl import Attribute

        shader_attributes = []
        for k, v in self.shader.shader_program._members.items():
            if isinstance(v, Attribute):
                shader_attributes.append(k)
        shader_attributes = filter_attributes(self.attributes, shader_attributes)

        vertex_buffer_object = self.shader.context.buffer(shader_attributes.tobytes())
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
            *shader_attributes.dtype.names,
            index_buffer=index_buffer_object,
        )
        vertex_array_object.render(self.primitive)
        vertex_buffer_object.release()
        vertex_array_object.release()
        if index_buffer_object is not None:
            index_buffer_object.release()


class Shader:
    def __init__(
        self,
        context,
        name=None,
        source=None,
    ):
        global shader_program_cache
        self.context = context
        self.name = name

        # See if the program is cached.
        if (
            self.name in shader_program_cache
            and shader_program_cache[self.name].ctx == self.context
        ):
            self.shader_program = shader_program_cache[self.name]
        elif source is not None:
            # Generate the shader from inline code if it was passed.
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
        if name is not None and name not in shader_program_cache:
            shader_program_cache[self.name] = self.shader_program

    def set_uniform(self, name, value):
        with contextlib.suppress(KeyError):
            self.shader_program[name] = value


class FullScreenQuad(Mesh):
    def __init__(
        self,
        context,
        fragment_shader_source=None,
        fragment_shader_name=None,
    ):
        if fragment_shader_source is None and fragment_shader_name is None:
            raise Exception("Must either pass shader name or shader source.")

        if fragment_shader_name is not None:
            # Use the name.
            shader_file_path = SHADER_FOLDER / f"{fragment_shader_name}.frag"
            fragment_shader_source = get_shader_code_from_file(shader_file_path)
        elif fragment_shader_source is not None:
            fragment_shader_source = textwrap.dedent(fragment_shader_source.lstrip())

        shader = Shader(
            context,
            source={
                "vertex_shader": """
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
                "fragment_shader": fragment_shader_source,
            },
        )
        attributes = np.zeros(6, dtype=[("in_vert", np.float32, (4,))])
        attributes["in_vert"] = np.array(
            [
                [-config["frame_x_radius"], -config["frame_y_radius"], 0, 1],
                [-config["frame_x_radius"], config["frame_y_radius"], 0, 1],
                [config["frame_x_radius"], config["frame_y_radius"], 0, 1],
                [-config["frame_x_radius"], -config["frame_y_radius"], 0, 1],
                [config["frame_x_radius"], -config["frame_y_radius"], 0, 1],
                [config["frame_x_radius"], config["frame_y_radius"], 0, 1],
            ],
        )
        shader.set_uniform("u_model_view_matrix", opengl.view_matrix())
        shader.set_uniform(
            "u_projection_matrix",
            opengl.orthographic_projection_matrix(),
        )
        super().__init__(shader, attributes)

    def render(self):
        super().render()
