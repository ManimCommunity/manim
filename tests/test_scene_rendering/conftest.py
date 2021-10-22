from pathlib import Path

import pytest

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
    with tempconfig(
        config.digest_file(Path(__file__).parent.parent / "standard_config.cfg"),
    ):
        config.media_dir = tmpdir
        yield


@pytest.fixture
def using_temp_opengl_config(tmpdir):
    """Standard fixture that makes tests use a standard_config.cfg with a temp dir."""
    with tempconfig(
        config.digest_file(Path(__file__).parent.parent / "standard_config.cfg"),
    ):
        config.media_dir = tmpdir
        config.renderer = "opengl"
        yield


@pytest.fixture
def disabling_caching():
    with tempconfig({"disable_caching": True}):
        yield


@pytest.fixture
def infallible_scenes_path():
    return str(Path(__file__).parent / "infallible_scenes.py")


@pytest.fixture
def use_opengl_renderer(enable_preview):
    with tempconfig({"renderer": "opengl", "preview": enable_preview}):
        yield


@pytest.fixture
def force_window_config_write_to_movie():
    with tempconfig({"force_window": True, "write_to_movie": True}):
        yield


@pytest.fixture
def force_window_config_pngs():
    with tempconfig({"force_window": True, "format": "png"}):
        yield
