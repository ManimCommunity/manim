"""Shader programs of the OpenGL renderer, compiled once per context."""

from __future__ import annotations

import contextlib
import re
from pathlib import Path
from typing import Any

import moderngl

SHADER_FOLDER = Path(__file__).parent / "shaders"
shader_program_cache: dict[str, moderngl.Program] = {}
file_path_to_code_map: dict[Path, str] = {}

__all__ = ["Shader"]


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


class Shader:
    def __init__(
        self,
        context: moderngl.Context,
        name: str | None = None,
        source: dict[str, Any] | None = None,
    ):
        global shader_program_cache
        self.context = context
        self.name = name
        self.source = source

        # See if the program is cached.
        if (
            self.name in shader_program_cache
            and shader_program_cache[self.name].ctx == self.context
        ):
            self.shader_program = shader_program_cache[self.name]
        elif self.source is not None:
            # Generate the shader from inline code if it was passed.
            self.shader_program = context.program(**self.source)
        elif self.name is not None:
            # Search for a file containing the shader.
            source_dict = {}
            source_dict_key = {
                "vert": "vertex_shader",
                "frag": "fragment_shader",
                "geom": "geometry_shader",
            }
            shader_folder = SHADER_FOLDER / self.name
            for shader_file in shader_folder.iterdir():
                shader_file_path = shader_folder / shader_file
                shader_source = get_shader_code_from_file(shader_file_path)
                source_dict[source_dict_key[shader_file_path.stem]] = shader_source
            self.shader_program = context.program(**source_dict)
        else:
            raise Exception("Must either pass shader name or shader source.")

        # Cache the shader.
        if self.name is not None and self.name not in shader_program_cache:
            shader_program_cache[self.name] = self.shader_program

    def set_uniform(self, name: str, value: Any) -> None:
        with contextlib.suppress(KeyError):
            self.shader_program[name] = value
