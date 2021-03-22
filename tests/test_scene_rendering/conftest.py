import pytest

from pathlib import Path

from manim import config, tempconfig


@pytest.fixture
def manim_cfg_file():
    return str(Path(__file__).parent / "manim.cfg")


@pytest.fixture
def simple_scenes_path():
    return str(Path(__file__).parent / "simple_scenes.py")


@pytest.fixture
def using_temp_config(tmpdir):
    """Standard fixture that makes tests use a standard_config.cfg with a temp dir."""
    with tempconfig(config.digest_file(Path(__file__).parent / "standard_config.cfg")):
        config.media_dir = tmpdir
        yield


@pytest.fixture
def disabling_caching():
    with tempconfig({"disable_caching": True}):
        yield


@pytest.fixture
def infallible_scenes_path():
    return str(Path(__file__).parent / "infallible_scenes.py")
