"""Tests for Table and related mobjects."""

from __future__ import annotations

import numpy as np

from manim import Circle, Table
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
