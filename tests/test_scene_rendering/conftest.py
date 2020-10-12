import pytest

from pathlib import Path

from manim import file_writer_config


@pytest.fixture
def manim_cfg_file():
    return Path(__file__).parent.joinpath("manim.cfg")


@pytest.fixture
def simple_scenes_path():
    return Path(__file__).parent.joinpath("simple_scenes.py")
