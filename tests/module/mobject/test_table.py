"""Tests for Table and related mobjects."""

from __future__ import annotations

from manim import Table
from manim.utils.color import GREEN


def test_highlighted_cell_color_access():
    """Test that accessing the color of a highlighted cell doesn't cause infinite recursion.

    Regression test for https://github.com/ManimCommunity/manim/issues/4419
    """
    table = Table([["This", "is a"], ["simple", "table"]])
    rect = table.get_highlighted_cell((1, 1), color=GREEN)

    # Should not raise RecursionError
    color = rect.color
    assert color == GREEN


def test_background_rectangle_color():
    """Test that BackgroundRectangle color access works correctly."""
    from manim.mobject.geometry.shape_matchers import BackgroundRectangle
    from manim import Square

    square = Square()
    bg_rect = BackgroundRectangle(square, color=GREEN)

    # Should not cause infinite recursion
    assert bg_rect.color == GREEN
