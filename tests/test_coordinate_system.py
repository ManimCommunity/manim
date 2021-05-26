import numpy as np
import pytest

from manim import LEFT, ORIGIN, Axes, ComplexPlane
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
