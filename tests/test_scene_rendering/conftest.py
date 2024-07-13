from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def manim_cfg_file():
    return Path(__file__).parent / "manim.cfg"


@pytest.fixture
def simple_scenes_path():
    return Path(__file__).parent / "simple_scenes.py"


@pytest.fixture
def standard_config(config):
    return config.digest_file(Path(__file__).parent.parent / "standard_config.cfg")


@pytest.fixture
def using_temp_config(tmpdir, standard_config):
    """Standard fixture that makes tests use a standard_config.cfg with a temp dir."""
    standard_config.media_dir = tmpdir


@pytest.fixture
def using_temp_opengl_config(tmpdir, standard_config, using_opengl_renderer):
    """Standard fixture that makes tests use a standard_config.cfg with a temp dir."""
    standard_config.media_dir = tmpdir


@pytest.fixture
def disabling_caching(config):
    config.disable_caching = True


@pytest.fixture
def infallible_scenes_path():
    return Path(__file__).parent / "infallible_scenes.py"


@pytest.fixture
def force_window_config_write_to_movie(config):
    config.force_window = True
    config.write_to_movie = True


@pytest.fixture
def force_window_config_pngs(config):
    config.force_window = True
    config.format = "png"
