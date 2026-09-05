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
def live_preview_config_movie(config):
    config.live_preview = True
    config.format = "mp4"


@pytest.fixture
def live_preview_config_pngs(config):
    config.live_preview = True
    config.format = "png-sequence"
