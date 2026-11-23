"""Paints are immutable values that vectorized mobjects are coloured with."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from manim import (
    BLUE,
    DOWN,
    GREEN,
    PI,
    RED,
    RIGHT,
    UL,
    LinearGradient,
    Solid,
    Square,
    Transform,
    interpolate_paint,
)
from manim.mobject.fields import field_version
from manim.paint import Paint
from manim.utils.space_ops import rotate_vector


def test_paints_cannot_be_changed():
    solid = Solid(RED, 0.5)
    gradient = LinearGradient([RED, BLUE])
    for paint in (solid, gradient):
        with pytest.raises(AttributeError):
            paint._rgbas = np.zeros((1, 4))
        with pytest.raises(ValueError, match="read-only"):
            paint.rgbas[0, 0] = 1
    with pytest.raises(ValueError, match="read-only"):
        gradient.direction[0] = 1


def test_a_paint_copies_what_it_is_given():
    opacities = np.array([0.2, 0.8])
    direction = np.array([1.0, 0.0, 0.0])
    gradient = LinearGradient([RED, BLUE], opacities, direction)
    opacities[0], direction[0] = 1, 0
    np.testing.assert_array_equal(gradient.opacities, [0.2, 0.8])
    np.testing.assert_array_equal(gradient.direction, RIGHT)


def test_gradients_default_to_up_left():
    np.testing.assert_array_equal(LinearGradient([RED, BLUE]).direction, UL)


def test_interpolation_keeps_the_paints_themselves():
    start, end = Solid(RED), LinearGradient([BLUE, GREEN], direction=RIGHT)
    assert interpolate_paint(start, end, 0) is start
    assert interpolate_paint(start, end, 1) is end
    assert interpolate_paint(start, start, 0.3) is start
    middle = interpolate_paint(start, end, 0.5)
    assert isinstance(middle, LinearGradient)
    np.testing.assert_allclose(
        middle.rgbas, (np.repeat(start.rgbas, 2, axis=0) + end.rgbas) / 2
    )
    # A solid takes the other paint's direction.
    np.testing.assert_array_equal(middle.direction, RIGHT)


def test_paints_of_unequal_length_are_stretched():
    three = LinearGradient([RED, GREEN, BLUE], opacities=[0, 0.5, 1])
    two = LinearGradient([BLUE, RED])
    middle = interpolate_paint(two, three, 0.5)
    assert len(middle.rgbas) == 3


def test_a_geometry_only_transform_changes_no_paint_version():
    square = Square().set_fill(RED, opacity=1)
    target = square.copy().shift(RIGHT).scale(0.5)
    versions = {name: field_version(square, name) for name in ("fill", "stroke")}
    animation = Transform(square, target)
    animation.begin()
    for alpha in (0.25, 0.5, 1):
        animation.interpolate(alpha)
    assert {name: field_version(square, name) for name in versions} == versions


def test_rotate_turns_gradients_and_leaves_solids():
    square = Square().set_fill([RED, BLUE], opacity=1)
    stroke = square.stroke
    square.rotate(PI / 2)
    np.testing.assert_allclose(square.fill.direction, rotate_vector(UL, PI / 2))
    assert square.stroke is stroke


def test_setters_keep_what_they_are_not_given():
    square = Square().set_fill([RED, BLUE], opacity=1).rotate(PI / 4)
    direction = square.fill.direction
    square.set_fill(opacity=0.5)
    np.testing.assert_array_equal(square.fill.direction, direction)
    assert square.get_fill_colors() == [RED, BLUE]
    square.set_fill(GREEN)
    np.testing.assert_allclose(square.get_fill_opacities(), [0.5, 0.5])


def test_setters_take_paints():
    gradient = LinearGradient([RED, BLUE], direction=DOWN)
    square = Square().set_fill(gradient)
    assert square.fill is gradient
    assert square.set_stroke(Solid(GREEN)).stroke.color == GREEN


def test_match_style_carries_the_paint():
    source = Square().set_fill(LinearGradient([RED, BLUE], direction=DOWN))
    target = Square().match_style(source)
    assert target.fill is source.fill


def test_copies_share_their_paints():
    square = Square().set_fill(RED, opacity=1)
    assert square.copy().fill is square.fill


_HASH = """
from manim import RED, Square, LinearGradient
from manim.utils.hashing import _get_json, _Memoizer
square = Square().set_fill(LinearGradient([RED, "#00FF00"]), opacity=1)
print(_get_json(square.fill, _Memoizer()))
"""


def test_paints_hash_by_content_in_any_process():
    from manim.utils.hashing import _get_json, _Memoizer

    def json(paint: Paint) -> str:
        return _get_json(paint, _Memoizer())

    assert json(Solid(RED)) == json(Solid(RED))
    assert json(Solid(RED)) != json(Solid(BLUE))
    assert json(LinearGradient([RED, BLUE])) != json(
        LinearGradient([RED, BLUE], direction=RIGHT)
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", _HASH], capture_output=True, text=True, check=True
        ).stdout
        for _ in range(2)
    }
    assert len(runs) == 1
