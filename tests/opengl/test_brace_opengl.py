from __future__ import annotations

import numpy as np

from manim import *


def test_brace_tip(config):
    brace = Brace(Square())
    # For the default DOWN brace the tip is its lowest point.
    assert np.allclose(brace.get_tip(), brace.get_bottom())
    assert np.array_equal(brace.get_tip(), brace.points[28])


def test_brace_tip_under_opengl(using_opengl_renderer):
    brace = Brace(Square())
    # For the default DOWN brace the tip is its lowest point.
    assert np.allclose(brace.get_tip(), brace.get_bottom())
    assert np.array_equal(brace.get_tip(), brace.points[35])
