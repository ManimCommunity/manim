from manim import config
import pytest
import numpy as np
import os
import sys
import logging
from shutil import rmtree


def pytest_addoption(parser):
    parser.addoption(
        "--skip_end_to_end",
        action="store_true",
        default=False,
        help="Will skip all the end-to-end tests. Useful when ffmpeg is not installed, e.g. on Windows jobs.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "skip_end_to_end: mark test as end_to_end test")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--skip_end_to_end"):
        return
    else:
        skip_end_to_end = pytest.mark.skip(
            reason="End to end test skipped due to --skip_end_to_end flag"
        )
        for item in items:
            if "skip_end_to_end" in item.keywords:
                item.add_marker(skip_end_to_end)


@pytest.fixture(scope="module")
def python_version():
    return "python3" if sys.platform == "darwin" else "python"


@pytest.fixture
def reset_cfg_file():
    cfgfilepath = os.path.join(os.path.dirname(__file__), "test_cli", "manim.cfg")
    with open(cfgfilepath) as cfgfile:
        original = cfgfile.read()
    yield
    with open(cfgfilepath, "w") as cfgfile:
        cfgfile.write(original)


@pytest.fixture
def clean_tests_cache():
    yield
    path_output = os.path.join("tests", "tests_cache", "media_temp")
    rmtree(path_output)
