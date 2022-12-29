from __future__ import annotations

import os

from manim._config import config
from manim.utils.file_ops import guarantee_existence


def get_directories() -> dict[str, str]:
    return config["directories"]


def get_temp_dir() -> str:
    return get_directories()["temporary_storage"]


def get_tex_dir() -> str:
    return guarantee_existence(os.path.join(get_temp_dir(), "Tex"))


def get_text_dir() -> str:
    return guarantee_existence(os.path.join(get_temp_dir(), "Text"))


def get_mobject_data_dir() -> str:
    return guarantee_existence(os.path.join(get_temp_dir(), "mobject_data"))


def get_downloads_dir() -> str:
    return guarantee_existence(os.path.join(get_temp_dir(), "manim_downloads"))


def get_output_dir() -> str:
    return guarantee_existence(get_directories()["output"])


def get_raster_image_dir() -> str:
    return get_directories()["raster_images"]


def get_vector_image_dir() -> str:
    return get_directories()["vector_images"]


def get_sound_dir() -> str:
    return get_directories()["sounds"]


def get_shader_dir() -> str:
    return get_directories()["shaders"]
