import numpy as np

from manim import BLUE, RED, Line
from manim.constants import LEFT, RIGHT


def _first_and_last_rgb(vmob):
    """Return RGB (no alpha) of first & last stroke rows."""
    rgbas = vmob.get_stroke_rgbas()
    return rgbas[0, :3], rgbas[-1, :3]


def test_gradient_left_to_right():
    """Blue should start at the left, red at the right."""
    seg = Line(LEFT, RIGHT).set_stroke([BLUE, RED], width=4)
    first, last = _first_and_last_rgb(seg)
    assert np.allclose(first, BLUE.to_rgb()), "Start of gradient is not BLUE"
    assert np.allclose(last, RED.to_rgb()), "End of gradient is not RED"


def test_gradient_right_to_left():
    """
    Reversing the points must **not** reverse the
    colour order supplied by the user.
    """
    seg = Line(RIGHT, LEFT).set_stroke([BLUE, RED], width=4)
    first, last = _first_and_last_rgb(seg)
    assert np.allclose(first, BLUE.to_rgb()), "Start of gradient is not BLUE"
    assert np.allclose(last, RED.to_rgb()), "End of gradient is not RED"
