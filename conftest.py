# This file is automatically picked by pytest
# while running tests. So, that each test is
# run on difference temporary directories and avoiding
# errors.

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
