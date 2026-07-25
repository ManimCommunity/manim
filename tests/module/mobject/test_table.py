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


def test_table_include_inner_lines_false():
    """Verify that inner lines can be disabled while outer lines remain."""
    table = Table(
        [["A", "B"], ["C", "D"]],
        include_outer_lines=True,
        include_inner_lines=False,
    )

    assert len(table.get_horizontal_lines()) == 2
    assert len(table.get_vertical_lines()) == 2


def test_table_include_inner_lines_true():
    """Verify that inner lines are present by default."""
    table = Table(
        [["A", "B"], ["C", "D"]],
        include_outer_lines=True,
        include_inner_lines=True,
    )

    assert len(table.get_horizontal_lines()) == 3
    assert len(table.get_vertical_lines()) == 3
