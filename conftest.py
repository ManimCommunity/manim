# This file is automatically picked by pytest
# while running tests. So, that each test is
# run on difference temporary directories and avoiding
# errors.

from __future__ import annotations

try:
    # https://github.com/moderngl/moderngl/issues/517
    import readline  # required to prevent a segfault on Python 3.10
except ModuleNotFoundError:  # windows
    pass

import moderngl

# If it is running Doctest the current directory
# is changed because it also tests the config module
# itself. If it's a normal test then it uses the
# tempconfig to change directories.
import pytest
from _pytest.doctest import DoctestItem

from manim import config, tempconfig


@pytest.fixture(autouse=True)
def temp_media_dir(tmpdir, monkeypatch, request):
    if isinstance(request.node, DoctestItem):
        monkeypatch.chdir(tmpdir)
        yield tmpdir
    else:
        with tempconfig({"media_dir": str(tmpdir)}):
            assert config.media_dir == str(tmpdir)
            yield tmpdir


def pytest_report_header(config):
    ctx = moderngl.create_standalone_context()
    info = ctx.info
    ctx.release()
    return (
        "\nOpenGL information",
        "------------------",
        f"vendor: {info['GL_VENDOR'].strip()}",
        f"renderer: {info['GL_RENDERER'].strip()}",
        f"version: {info['GL_VERSION'].strip()}\n",
    )
