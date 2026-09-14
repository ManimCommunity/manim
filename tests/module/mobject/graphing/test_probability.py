from __future__ import annotations

from manim import BarChart, DecimalNumber, VGroup


def _number_groups(axis):
    """Return the groups of number labels attached to ``axis``."""
    return [
        mob
        for mob in axis.submobjects
        if isinstance(mob, VGroup)
        and len(mob) > 0
        and all(isinstance(sub, DecimalNumber) for sub in mob)
    ]


def test_barchart_y_axis_include_numbers_false():
    chart = BarChart(
        values=[1, 2, 3],
        y_axis_config={"include_numbers": False},
    )
    assert _number_groups(chart.y_axis) == []


def test_barchart_y_axis_numbers_are_added_once():
    # Keep the default label constructor: DecimalNumber caches digit mobjects
    # globally by string, so a non-default one here leaks into later tests.
    chart = BarChart(
        values=[1, 2, 3],
        y_axis_config={"include_numbers": True},
    )
    assert len(_number_groups(chart.y_axis)) == 1
