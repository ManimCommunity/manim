from __future__ import annotations

import numpy as np
import pytest

from manim import *


def test_brace_tip(config):
    brace = Brace(Square())
    # For the default DOWN brace the tip is its lowest point.
    assert np.allclose(brace.get_tip(), brace.get_bottom())


@pytest.mark.parametrize(
    ("width", "path_config"),
    [
        (2, {}),
        (2, {"long_lines": True}),
        (2, {"should_subdivide_sharp_curves": True}),
        (0.1, {"should_remove_null_curves": True}),
        (
            0.1,
            {
                "long_lines": True,
                "should_subdivide_sharp_curves": True,
                "should_remove_null_curves": True,
            },
        ),
    ],
    ids=["default", "long-lines", "subdivided", "null-curves", "all-flags"],
)
def test_brace_tip_under_opengl(using_opengl_renderer, width, path_config):
    brace = Brace(Rectangle(width=width, height=1), **path_config)
    # For a DOWN brace the tip is its lowest point.
    assert np.allclose(brace.get_tip(), brace.get_bottom())
