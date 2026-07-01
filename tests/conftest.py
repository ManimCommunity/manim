from __future__ import annotations

import logging
import sys
from pathlib import Path

import cairo
import moderngl
import pytest

import manim


def pytest_report_header(config):
    try:
        ctx = moderngl.create_standalone_context()
        info = ctx.info
        ctx.release()
    except Exception as e:
        raise Exception("Error while creating moderngl context") from e

    return (
        f"\nCairo Version: {cairo.cairo_version()}",
        "\nOpenGL information",
        "------------------",
        f"vendor: {info['GL_VENDOR'].strip()}",
        f"renderer: {info['GL_RENDERER'].strip()}",
        f"version: {info['GL_VERSION'].strip()}\n",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--skip_slow",
        action="store_true",
        default=False,
        help="Will skip all the slow marked tests. Slow tests are arbitrarily marked as such.",
    )
    parser.addoption(
        "--show_diff",
        action="store_true",
        default=False,
        help="Will show a visual comparison if a graphical unit test fails.",
    )
    parser.addoption(
        "--set_test",
        action="store_true",
        default=False,
        help="Will create the control data for EACH running tests. ",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "skip_end_to_end: mark test as end_to_end test")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--skip_slow"):
        return
    else:
        slow_skip = pytest.mark.skip(
            reason="Slow test skipped due to --disable_slow flag.",
        )
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(slow_skip)


@pytest.fixture(autouse=True)
def temp_media_dir(tmpdir, monkeypatch, request):
    if isinstance(request.node, pytest.DoctestItem):
        monkeypatch.chdir(tmpdir)
        yield tmpdir
    else:
        with manim.tempconfig({"media_dir": str(tmpdir)}):
            assert manim.config.media_dir == str(tmpdir)
            yield tmpdir


@pytest.fixture
def manim_caplog(caplog):
    logger = logging.getLogger("manim")
    logger.propagate = True
    caplog.set_level(logging.INFO, logger="manim")
    yield caplog
    logger.propagate = False


@pytest.fixture
def config():
    saved = manim.config.copy()
    manim.config.renderer = "cairo"
    # we need to return the actual config so that tests
    # using tempconfig pass
    yield manim.config
    manim.config.update(saved)


@pytest.fixture
def dry_run(config):
    config.dry_run = True


@pytest.fixture(scope="session")
def python_version():
    # use the same python executable as it is running currently
    # rather than randomly calling using python or python3, which
    # may create problems.
    return sys.executable


@pytest.fixture
def reset_cfg_file():
    cfgfilepath = Path(__file__).parent / "test_cli" / "manim.cfg"
    original = cfgfilepath.read_text()
    yield
    cfgfilepath.write_text(original)


@pytest.fixture
def using_opengl_renderer(config):
    """Standard fixture for running with opengl that makes tests use a standard_config.cfg with a temp dir."""
    config.renderer = "opengl"
    yield
    # as a special case needed to manually revert back to cairo
    # due to side effects of setting the renderer
    config.renderer = "cairo"
