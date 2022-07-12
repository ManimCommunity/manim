from __future__ import annotations

import numpy as np

from manim import NumberLine
from manim.mobject.text.numbers import Integer


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


def test_number_to_point():
    line = NumberLine()
    numbers = [1, 2, 3, 4, 5]
    numbers_np = np.array(numbers)
    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ]
    )
    vec_1 = np.array([line.number_to_point(x) for x in numbers])
    vec_2 = line.number_to_point(numbers)
    vec_3 = line.number_to_point(numbers_np)

    np.testing.assert_equal(
        np.round(vec_1, 4),
        np.round(expected, 4),
        f"Expected {expected} but got {vec_1} with input as scalar",
    )
    np.testing.assert_equal(
        np.round(vec_2, 4),
        np.round(expected, 4),
        f"Expected {expected} but got {vec_2} with input as params",
    )
    np.testing.assert_equal(
        np.round(vec_2, 4),
        np.round(expected, 4),
        f"Expected {expected} but got {vec_3} with input as ndarray",
    )


def test_point_to_number():
    line = NumberLine()
    points = [
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ]
    points_np = np.array(points)
    expected = [1, 2, 3, 4, 5]

    num_1 = [line.point_to_number(point) for point in points]
    num_2 = line.point_to_number(points)
    num_3 = line.point_to_number(points_np)

    np.testing.assert_array_equal(np.round(num_1, 4), np.round(expected, 4))
    np.testing.assert_array_equal(np.round(num_2, 4), np.round(expected, 4))
    np.testing.assert_array_equal(np.round(num_3, 4), np.round(expected, 4))
