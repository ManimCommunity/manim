from __future__ import annotations

import pytest


@pytest.fixture
def show_diff(request):
    return request.config.getoption("show_diff")


@pytest.fixture(params=[True, False])
def use_vectorized(request):
    yield request.param
