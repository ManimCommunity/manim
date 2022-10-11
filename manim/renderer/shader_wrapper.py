from __future__ import annotations

import copy
import re
from pathlib import Path

import moderngl
import numpy as np

from .. import logger

# Mobjects that should be rendered with
# the same shader will be organized and
# clumped together based on keeping track
# of a dict holding all the relevant information
# to that shader


def get_shader_dir():
    return Path(__file__).parent / "shaders"


def find_file(file_name: Path, directories: list[Path]) -> Path:
    # Check if what was passed in is already a valid path to a file
    if file_name.exists():
        return file_name
    possible_paths = (directory / file_name for directory in directories)
    for path in possible_paths:
        if path.exists():
            return path
        else:
            logger.debug(f"{path} does not exist.")
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
            return get_shader_code_from_file(
                self.shader_folder / f"{name}.glsl",
            )

        self.program_code = {
            "vertex_shader": get_code("vert"),
            "geometry_shader": get_code("geom"),
            "fragment_shader": get_code("frag"),
        }

    def get_program_code(self):
        return self.program_code

    def replace_code(self, old, new):
        code_map = self.program_code
        for (name, _code) in code_map.items():
            if code_map[name] is None:
                continue
            code_map[name] = re.sub(old, new, code_map[name])
        self.refresh_id()

    def combine_with(self, *shader_wrappers):
        # Assume they are of the same type
        if len(shader_wrappers) == 0:
            return
        if self.vert_indices is not None:
            num_verts = len(self.vert_data)
            indices_list = [self.vert_indices]
            data_list = [self.vert_data]
            for sw in shader_wrappers:
                indices_list.append(sw.vert_indices + num_verts)
                data_list.append(sw.vert_data)
                num_verts += len(sw.vert_data)
            self.vert_indices = np.hstack(indices_list)
            self.vert_data = np.hstack(data_list)
        else:
            self.vert_data = np.hstack(
                [self.vert_data, *(sw.vert_data for sw in shader_wrappers)],
            )
        return self


# For caching
filename_to_code_map: dict = {}


def get_shader_code_from_file(filename: Path) -> str | None:
    if filename in filename_to_code_map:
        return filename_to_code_map[filename]

    try:
        filepath = find_file(
            filename,
            directories=[get_shader_dir(), Path("/")],
        )
    except OSError:
        return None

    result = filepath.read_text()

    # To share functionality between shaders, some functions are read in
    # from other files an inserted into the relevant strings before
    # passing to ctx.program for compiling
    # Replace "#INSERT " lines with relevant code
    insertions = re.findall(
        r"^#include ../include/.*\.glsl$",
        result,
        flags=re.MULTILINE,
    )
    for line in insertions:
        inserted_code = get_shader_code_from_file(
            Path() / "include" / line.replace("#include ../include/", ""),
        )
        if inserted_code is None:
            return None
        result = result.replace(line, inserted_code)
    filename_to_code_map[filename] = result
    return result


def get_colormap_code(rgb_list):
    data = ",".join("vec3({}, {}, {})".format(*rgb) for rgb in rgb_list)
    return f"vec3[{len(rgb_list)}]({data})"
