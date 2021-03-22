import pytest

from pathlib import Path


@pytest.fixture
def manim_cfg_file():
    return str(Path(__file__).parent / "manim.cfg")


@pytest.fixture
def simple_scenes_path():
    return str(Path(__file__).parent / "simple_scenes.py")


@pytest.fixture
def infallible_scenes_path():
    return str(Path(__file__).parent / "infallible_scenes.py")
