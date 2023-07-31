from __future__ import annotations

import pytest


@pytest.fixture
def show_diff(request):
    return request.config.getoption("show_diff")
