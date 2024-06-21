# For caching
import re
from functools import lru_cache
from pathlib import Path

import moderngl as gl

from manim._config import logger

filename_to_code_map: dict = {}


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
        logger.warning(f"Could not find shader file {filename}")
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


def load_shader_program_by_folder(ctx: gl.Context, folder_name: str):
    vertex_code = get_shader_code_from_file(Path(folder_name + "/vert.glsl"))
    geometry_code = get_shader_code_from_file(Path(folder_name + "/geom.glsl"))
    fragment_code = get_shader_code_from_file(Path(folder_name + "/frag.glsl"))
    # print(folder_name)
    # for i,l in enumerate(geometry_code.splitlines()):
    #     print(str(i) + ":" + l )
    if vertex_code is None or fragment_code is None:
        logger.error(
            f"Invalid program definition for {folder_name} vertex or fragment shader not present"
        )
        raise RuntimeError("Loading Shader Program Error")
    if geometry_code is None:
        return ctx.program(vertex_shader=vertex_code, fragment_shader=fragment_code)
    return ctx.program(
        vertex_shader=vertex_code,
        geometry_shader=geometry_code,
        fragment_shader=fragment_code,
    )
