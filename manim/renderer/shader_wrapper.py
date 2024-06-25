from __future__ import annotations

import copy
import re
from functools import lru_cache
from pathlib import Path

import moderngl
import numpy as np

from manim.utils.iterables import resize_array

from .. import logger

# Mobjects that should be rendered with
# the same shader will be organized and
# clumped together based on keeping track
# of a dict holding all the relevant information
# to that shader

__all__ = ["ShaderWrapper"]


def get_shader_dir():
    return Path(__file__).parent / "shaders"


def find_file(file_name: Path, directories: list[Path]) -> Path:
    # Check if what was passed in is already a valid path to a file
    if file_name.exists():
        return file_name
    possible_paths = (directory / file_name for directory in directories)
    for path in possible_paths:
        logger.debug(f"Searching for {file_name} in {path}")
        if path.exists():
            return path
        else:
            logger.debug(f"shader_wrapper.py::find_file() : {path} does not exist.")
    raise OSError(f"{file_name} not Found")


class ShaderWrapper:
    def __init__(
        self,
        vert_data=None,
        vert_indices=None,
        shader_folder=None,
        uniforms=None,  # A dictionary mapping names of uniform variables
        texture_paths=None,  # A dictionary mapping names to filepaths for textures.
        depth_test=False,
        render_primitive=moderngl.TRIANGLE_STRIP,
    ):
        self.vert_data = vert_data
        self.vert_indices = vert_indices
        self.vert_attributes = vert_data.dtype.names
        self.shader_folder = Path(shader_folder or "")
        self.uniforms = uniforms or {}
        self.texture_paths = texture_paths or {}
        self.depth_test = depth_test
        self.render_primitive = str(render_primitive)
        self.init_program_code()
        self.refresh_id()

    def __eq__(self, shader_wrapper: object):
        if not isinstance(shader_wrapper, ShaderWrapper):
            raise TypeError(
                f"Cannot compare ShaderWrapper with non-ShaderWrapper object of type {type(shader_wrapper)}"
            )
        return all(
            (
                np.all(self.vert_data == shader_wrapper.vert_data),
                np.all(self.vert_indices == shader_wrapper.vert_indices),
                self.shader_folder == shader_wrapper.shader_folder,
                all(
                    np.all(self.uniforms[key] == shader_wrapper.uniforms[key])
                    for key in self.uniforms
                ),
                all(
                    self.texture_paths[key] == shader_wrapper.texture_paths[key]
                    for key in self.texture_paths
                ),
                self.depth_test == shader_wrapper.depth_test,
                self.render_primitive == shader_wrapper.render_primitive,
            )
        )

    def copy(self):
        result = copy.copy(self)
        result.vert_data = np.array(self.vert_data)
        if result.vert_indices is not None:
            result.vert_indices = np.array(self.vert_indices)
        if self.uniforms:
            result.uniforms = dict(self.uniforms)
        if self.texture_paths:
            result.texture_paths = dict(self.texture_paths)
        return result

    def is_valid(self):
        return all(
            [
                self.vert_data is not None,
                self.program_code["vertex_shader"] is not None,
                self.program_code["fragment_shader"] is not None,
            ],
        )

    def get_id(self):
        return self.id

    def get_program_id(self):
        return self.program_id

    def create_id(self):
        # A unique id for a shader
        return "|".join(
            map(
                str,
                [
                    self.program_id,
                    self.uniforms,
                    self.texture_paths,
                    self.depth_test,
                    self.render_primitive,
                ],
            ),
        )

    def refresh_id(self):
        self.program_id = self.create_program_id()
        self.id = self.create_id()

    def create_program_id(self):
        return hash(
            "".join(
                self.program_code[f"{name}_shader"] or ""
                for name in ("vertex", "geometry", "fragment")
            ),
        )

    def init_program_code(self):
        def get_code(name: str) -> str | None:
            path = self.shader_folder / f"{name}.glsl"
            logger.debug(f"Reading {name}.glsl shader code from {path.absolute()}")
            code = get_shader_code_from_file(path)
            if code is not None:
                logger.debug(
                    f"=============================================\n{code}\n============================================="
                )
            return code

        self.program_code = {
            "vertex_shader": get_code("vert"),
            "geometry_shader": get_code("geom"),
            "fragment_shader": get_code("frag"),
        }

    def get_program_code(self):
        return self.program_code

    def replace_code(self, old, new):
        code_map = self.program_code
        for name, _code in code_map.items():
            if code_map[name] is None:
                continue
            code_map[name] = re.sub(old, new, code_map[name])
        self.refresh_id()

    def combine_with(self, *shader_wrappers: ShaderWrapper) -> ShaderWrapper:
        self.read_in(self.copy(), *shader_wrappers)
        return self

    def read_in(self, *shader_wrappers: ShaderWrapper) -> ShaderWrapper:
        # Assume all are of the same type
        total_len = sum(len(sw.vert_data) for sw in shader_wrappers)
        self.vert_data = resize_array(self.vert_data, total_len)
        if self.vert_indices is not None:
            total_verts = sum(len(sw.vert_indices) for sw in shader_wrappers)
            self.vert_indices = resize_array(self.vert_indices, total_verts)

        n_points = 0
        n_verts = 0
        for sw in shader_wrappers:
            new_n_points = n_points + len(sw.vert_data)
            self.vert_data[n_points:new_n_points] = sw.vert_data
            if self.vert_indices is not None and sw.vert_indices is not None:
                new_n_verts = n_verts + len(sw.vert_indices)
                self.vert_indices[n_verts:new_n_verts] = sw.vert_indices + n_points
                n_verts = new_n_verts
            n_points = new_n_points
        return self


# For caching
filename_to_code_map: dict = {}


@lru_cache(maxsize=12)
def get_shader_code_from_file(filename: Path) -> str | None:
    if filename in filename_to_code_map:
        return filename_to_code_map[filename]
    try:
        filepath = find_file(
            filename,
            directories=[get_shader_dir(), Path("/")],
        )
    except OSError:
        logger.warning(f"Could not find shader file {filename.absolute()}")
        return None

    result = filepath.read_text()

    # To share functionality between shaders, some functions are read in
    # from other files an inserted into the relevant strings before
    # passing to ctx.program for compiling
    # Replace "#INSERT " lines with relevant code
    insertions = re.findall(
        r"^#include.*",
        result,
        flags=re.MULTILINE,
    )
    for line in insertions:
        include_path = line.strip().replace("#include", "")
        include_path = include_path.replace('"', "")
        path = (filepath.parent / Path(include_path.strip())).resolve()
        logger.debug(f"Trying to get code from: {path} to include in {filepath.name}")
        inserted_code = get_shader_code_from_file(
            path,
        )
        if inserted_code is None:
            return None

        result = result.replace(
            line,
            f"// Start include of: {include_path}\n\n{inserted_code}\n\n// End include of: {include_path}",
        )
    filename_to_code_map[filename] = result
    return result


def get_colormap_code(rgb_list):
    data = ",".join("vec3({}, {}, {})".format(*rgb) for rgb in rgb_list)
    return f"vec3[{len(rgb_list)}]({data})"
