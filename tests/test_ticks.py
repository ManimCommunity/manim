import numpy as np

from manim import PI, Axes, NumberLine


def test_duplicate_ticks_removed_for_axes():
    axis = NumberLine(
        x_range=[-10, 10],
    )
    ticks = axis.get_tick_range()
    assert np.unique(ticks).size == ticks.size


def test_ticks_not_generated_on_origin_for_axes():
    axes = Axes(
        x_range=[-10, 10],
        y_range=[-10, 10],
        axis_config={"include_ticks": True},
    )

    x_axis_range = axes.x_axis.get_tick_range()
    y_axis_range = axes.y_axis.get_tick_range()

    assert 0 not in x_axis_range
    assert 0 not in y_axis_range


def test_expected_ticks_generated():
    axes = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_ticks": True})
    x_axis_range = axes.x_axis.get_tick_range()
    y_axis_range = axes.y_axis.get_tick_range()

    assert 1 in x_axis_range
    assert 1 in y_axis_range
    assert -1 in x_axis_range
    assert -1 in y_axis_range


def test_ticks_generated_from_origin_for_axes():
    axes = Axes(
        x_range=[-PI, PI],
        y_range=[-PI, PI],
        axis_config={"include_ticks": True},
    )
    x_axis_range = axes.x_axis.get_tick_range()
    y_axis_range = axes.y_axis.get_tick_range()

    assert -2 in x_axis_range
    assert -1 in x_axis_range
    assert 0 not in x_axis_range
    assert 1 in x_axis_range
    assert 2 in x_axis_range

    assert -2 in y_axis_range
    assert -1 in y_axis_range
    assert 0 not in y_axis_range
    assert 1 in y_axis_range
    assert 2 in y_axis_range
