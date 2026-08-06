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


def test_table_col_widths_are_honored():
    """Fixed ``col_widths`` should be applied to every column, including the
    first and last, instead of being truncated to the cell contents.

    Regression test for https://github.com/ManimCommunity/manim/issues/3446
    """
    table = Table(
        [["1", "10", "100", "1000"], ["0", "0", "0", "0"]],
        h_buff=0.1,
        v_buff=0.1,
        include_outer_lines=True,
        arrange_in_grid_config={"col_widths": [3] * 4, "col_alignments": ["c"] * 4},
    )
    line_xs = sorted(line.get_center()[0] for line in table.get_vertical_lines())
    drawn_widths = [line_xs[i + 1] - line_xs[i] for i in range(len(line_xs) - 1)]

    # All columns, outer ones included, must have the same drawn width.
    assert max(drawn_widths) - min(drawn_widths) < 1e-6
    # Each drawn column spans its slot width plus the horizontal buffer.
    for width in drawn_widths:
        assert abs(width - (3 + 0.1)) < 1e-6
