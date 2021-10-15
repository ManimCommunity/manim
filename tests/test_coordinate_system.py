import math

import numpy as np
import pytest

from manim import LEFT, ORIGIN, PI, UR, Axes, Circle, ComplexPlane
from manim import CoordinateSystem as CS
from manim import NumberPlane, PolarPlane, ThreeDAxes, config, tempconfig


def test_initial_config():
    """Check that all attributes are defined properly from the config."""
    cs = CS()
    assert cs.x_range[0] == round(-config["frame_x_radius"])
    assert cs.x_range[1] == round(config["frame_x_radius"])
    assert cs.x_range[2] == 1.0
    assert cs.y_range[0] == round(-config["frame_y_radius"])
    assert cs.y_range[1] == round(config["frame_y_radius"])
    assert cs.y_range[2] == 1.0

    ax = Axes()
    assert np.allclose(ax.get_center(), ORIGIN)
    assert np.allclose(ax.y_axis_config["label_direction"], LEFT)

    with tempconfig({"frame_x_radius": 100, "frame_y_radius": 200}):
        cs = CS()
        assert cs.x_range[0] == -100
        assert cs.x_range[1] == 100
        assert cs.y_range[0] == -200
        assert cs.y_range[1] == 200


def test_dimension():
    """Check that objects have the correct dimension."""
    assert Axes().dimension == 2
    assert NumberPlane().dimension == 2
    assert PolarPlane().dimension == 2
    assert ComplexPlane().dimension == 2
    assert ThreeDAxes().dimension == 3


def test_abstract_base_class():
    """Check that CoordinateSystem has some abstract methods."""
    with pytest.raises(Exception):
        CS().get_axes()


def test_NumberPlane():
    """Test that NumberPlane generates the correct number of lines when its ranges do not cross 0."""
    pos_x_range = (0, 7)
    neg_x_range = (-7, 0)

    pos_y_range = (2, 6)
    neg_y_range = (-6, -2)

    x_vals = [0, 1.5, 2, 2.8, 4, 6.25]
    y_vals = [2, 5, 4.25, 6, 4.5, 2.75]

    testing_data = [
        (pos_x_range, pos_y_range, x_vals, y_vals),
        (pos_x_range, neg_y_range, x_vals, [-v for v in y_vals]),
        (neg_x_range, pos_y_range, [-v for v in x_vals], y_vals),
        (neg_x_range, neg_y_range, [-v for v in x_vals], [-v for v in y_vals]),
    ]

    for test_data in testing_data:

        x_range, y_range, x_vals, y_vals = test_data

        x_start, x_end = x_range
        y_start, y_end = y_range

        plane = NumberPlane(
            x_range=x_range,
            y_range=y_range,
            # x_length = 7,
            axis_config={"include_numbers": True},
        )

        # normally these values would be need to be added by one to pass since there's an
        # overlapping pair of lines at the origin, but since these planes do not cross 0,
        # this is not needed.
        num_y_lines = math.ceil(x_end - x_start)
        num_x_lines = math.floor(y_end - y_start)

        assert len(plane.y_lines) == num_y_lines
        assert len(plane.x_lines) == num_x_lines

    plane = NumberPlane((-5, 5, 0.5), (-8, 8, 2))  # <- test for different step values
    assert len(plane.x_lines) == 8
    assert len(plane.y_lines) == 20


def test_point_to_coords():
    ax = Axes(x_range=[0, 10, 2])
    circ = Circle(radius=0.5).shift(UR * 2)

    # get the coordinates of the circle with respect to the axes
    coords = np.around(ax.point_to_coords(circ.get_right()), decimals=4)
    assert np.array_equal(coords, (7.0833, 2.6667))


def test_coords_to_point():
    ax = Axes()

    # a point with respect to the axes
    c2p_coord = np.around(ax.coords_to_point(2, 2), decimals=4)
    assert np.array_equal(c2p_coord, (1.7143, 1.5, 0))


def test_input_to_graph_point():
    ax = Axes()
    curve = ax.get_graph(lambda x: np.cos(x))
    line_graph = ax.get_line_graph([1, 3, 5], [-1, 2, -2], add_vertex_dots=False)[
        "line_graph"
    ]

    # move a square to PI on the cosine curve.
    position = np.around(ax.input_to_graph_point(x=PI, graph=curve), decimals=4)
    assert np.array_equal(position, (2.6928, -0.75, 0))

    # test the line_graph implementation
    position = np.around(ax.input_to_graph_point(x=PI, graph=line_graph), decimals=4)
    assert np.array_equal(position, (2.6928, 1.2876, 0))
