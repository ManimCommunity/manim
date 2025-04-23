"""
Unit tests for #4198 – gradient order must be stable no matter how the line
is oriented.  The test is intentionally **numeric only** (no rendering), so it
runs quickly under pytest-xdist.
"""

from manim import BLUE, LEFT, RED, RIGHT, Line


def _first_last_stroke_colors(vmob):
    """Return the first and last stroke colours as ManimColor objects."""
    cols = vmob.get_stroke_colors()
    # For a two-stop gradient Manim stores exactly two rows in stroke_rgbas
    assert len(cols) >= 2, "gradient expected at least two colour stops"
    return cols[0], cols[-1]


def test_gradient_order_preserved_forward():
    """Baseline sanity: forward-oriented line keeps [BLUE, RED]."""
    line = Line(LEFT, RIGHT).set_color([BLUE, RED])
    first, last = _first_last_stroke_colors(line)
    assert first == BLUE
    assert last == RED


def test_gradient_order_preserved_reversed():
    """
    Reversed anchor order used to flip the gradient; the fix in PR #4227 should
    keep the user-specified order intact.
    """
    line = Line(RIGHT, LEFT).set_color([BLUE, RED])
    first, last = _first_last_stroke_colors(line)
    assert first == BLUE
    assert last == RED
