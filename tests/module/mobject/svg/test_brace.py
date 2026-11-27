import numpy as np

from manim import Brace, Square


def test_brace_tip():
    brace = Brace(Square())
    # For the default DOWN brace the tip is its lowest point.
    assert np.allclose(brace.get_tip(), brace.get_bottom())
