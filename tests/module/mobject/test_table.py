"""Tests for Table and related mobjects."""

from __future__ import annotations

import numpy as np

from manim import LEFT, Circle, Table, Text
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


def test_table_col_widths_with_labels():
    """``col_widths`` should be honored even when row/column labels are present.

    Regression test for https://github.com/ManimCommunity/manim/issues/3446
    """
    table = Table(
        [["a", "b"], ["c", "d"]],
        row_labels=[Text("r1"), Text("r2")],
        col_labels=[Text("c1"), Text("c2")],
        top_left_entry=Text(""),
        h_buff=0.1,
        include_outer_lines=True,
        arrange_in_grid_config={"col_widths": [2, 3, 3], "col_alignments": "ccc"},
    )
    line_xs = sorted(line.get_center()[0] for line in table.get_vertical_lines())
    drawn_widths = [line_xs[i + 1] - line_xs[i] for i in range(len(line_xs) - 1)]
    for width, expected in zip(drawn_widths, [2 + 0.1, 3 + 0.1, 3 + 0.1], strict=True):
        assert abs(width - expected) < 1e-6


def test_table_col_widths_respects_cell_alignment():
    """A non-centered ``cell_alignment`` must not be treated as centered when
    positioning the grid lines around a fixed-width column.

    Regression test for https://github.com/ManimCommunity/manim/issues/3446
    """
    table = Table(
        [["a"]],
        h_buff=0.1,
        include_outer_lines=True,
        arrange_in_grid_config={"col_widths": [3], "cell_alignment": LEFT},
    )
    column = table.get_columns()[0]
    actual_left = min(line.get_center()[0] for line in table.get_vertical_lines())
    expected_left = column.get_left()[0] - table.h_buff / 2
    assert abs(actual_left - expected_left) < 1e-6


def test_table_accepts_iterable_data():
    """Table data can be any iterable of iterables."""
    data = (iter(row) for row in [["A", "B"], ["C", "D"]])

    table = Table(data)

    assert len(table.get_entries()) == 4


def test_table_accepts_numpy_label_arrays():
    """NumPy arrays remain valid iterables for row and column labels."""
    for label_parameter, label_getter in (
        ("row_labels", Table.get_row_labels),
        ("col_labels", Table.get_col_labels),
    ):
        labels = np.empty(2, dtype=object)
        labels[:] = [Circle(), Circle()]

        table = Table(
            [["A", "B"], ["C", "D"]],
            **{label_parameter: labels},
        )
        actual_labels = label_getter(table)

        assert len(actual_labels) == len(labels)
        assert all(
            actual_labels[index] is labels[index] for index in range(len(labels))
        )


def test_empty_label_iterables_are_treated_as_absent():
    """Empty label iterables do not affect the other label dimension."""
    col_labels = [Circle(), Circle()]
    table_without_row_labels = Table(
        [["A", "B"], ["C", "D"]],
        row_labels=iter(()),
        col_labels=col_labels,
    )

    assert len(table_without_row_labels.get_row_labels()) == 0
    assert len(table_without_row_labels.get_col_labels()) == 2

    row_labels = [Circle(), Circle()]
    table_without_col_labels = Table(
        [["A", "B"], ["C", "D"]],
        row_labels=row_labels,
        col_labels=iter(()),
    )

    assert len(table_without_col_labels.get_row_labels()) == 2
    assert len(table_without_col_labels.get_col_labels()) == 0
