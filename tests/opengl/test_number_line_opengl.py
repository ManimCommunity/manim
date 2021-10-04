import numpy as np

from manim import NumberLine
from manim.mobject.numbers import Integer


def test_unit_vector():
    """Check if the magnitude of unit vector along
    the NumberLine is equal to its unit_size."""
    axis1 = NumberLine(unit_size=0.4)
    axis2 = NumberLine(x_range=[-2, 5], length=12)
    for axis in (axis1, axis2):
        assert np.linalg.norm(axis.get_unit_vector()) == axis.unit_size


def test_decimal_determined_by_step():
    """Checks that step size is considered when determining the number of decimal
    places."""
    axis = NumberLine(x_range=[-2, 2, 0.5])
    expected_decimal_places = 1
    actual_decimal_places = axis.decimal_number_config["num_decimal_places"]
    assert actual_decimal_places == expected_decimal_places, (
        "Expected 1 decimal place but got " + actual_decimal_places
    )

    axis2 = NumberLine(x_range=[-1, 1, 0.25])
    expected_decimal_places = 2
    actual_decimal_places = axis2.decimal_number_config["num_decimal_places"]
    assert actual_decimal_places == expected_decimal_places, (
        "Expected 1 decimal place but got " + actual_decimal_places
    )


def test_decimal_config_overrides_defaults():
    """Checks that ``num_decimal_places`` is determined by step size and gets overridden by ``decimal_number_config``."""
    axis = NumberLine(
        x_range=[-2, 2, 0.5],
        decimal_number_config={"num_decimal_places": 0},
    )
    expected_decimal_places = 0
    actual_decimal_places = axis.decimal_number_config["num_decimal_places"]
    assert actual_decimal_places == expected_decimal_places, (
        "Expected 1 decimal place but got " + actual_decimal_places
    )


def test_whole_numbers_step_size_default_to_0_decimal_places():
    """Checks that ``num_decimal_places`` defaults to 0 when a whole number step size is passed."""
    axis = NumberLine(x_range=[-2, 2, 1])
    expected_decimal_places = 0
    actual_decimal_places = axis.decimal_number_config["num_decimal_places"]
    assert actual_decimal_places == expected_decimal_places, (
        "Expected 1 decimal place but got " + actual_decimal_places
    )


def test_add_labels():
    expected_label_length = 6
    num_line = NumberLine(x_range=[-4, 4])
    num_line.add_labels(
        dict(zip(list(range(-3, 3)), [Integer(m) for m in range(-1, 5)])),
    )
    actual_label_length = len(num_line.labels)
    assert (
        actual_label_length == expected_label_length
    ), f"Expected a VGroup with {expected_label_length} integers but got {actual_label_length}."
