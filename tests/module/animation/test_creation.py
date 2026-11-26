from __future__ import annotations

import numpy as np
import pytest

from manim import (
    AddTextLetterByLetter,
    Rectangle,
    RemoveTextLetterByLetter,
    ShowIncreasingSubsets,
    ShowSubmobjectsOneByOne,
    Square,
    Text,
    TypeWithCursor,
    VGroup,
)


def test_non_empty_text_creation():
    """Check if AddTextLetterByLetter works for non-empty text."""
    s = Text("Hello")
    anim = AddTextLetterByLetter(s)
    assert anim.mobject.text == "Hello"


def test_empty_text_creation():
    """Ensure ValueError is raised for empty text."""
    with pytest.raises(ValueError, match="does not seem to contain any characters"):
        AddTextLetterByLetter(Text(""))


def test_whitespace_text_creation():
    """Ensure ValueError is raised for whitespace-only text, assuming the whitespace characters have no points."""
    with pytest.raises(ValueError, match="does not seem to contain any characters"):
        AddTextLetterByLetter(Text("    "))


def test_run_time_for_non_empty_text(config):
    """Ensure the run_time is calculated correctly for non-empty text."""
    s = Text("Hello")
    run_time_per_char = 0.1
    expected_run_time = np.max((1 / config.frame_rate, run_time_per_char)) * len(s.text)
    anim = AddTextLetterByLetter(s, time_per_char=run_time_per_char)
    assert anim.run_time == expected_run_time


def opacities(mobjects):
    return [
        (round(mob.get_fill_opacity(), 6), round(mob.get_stroke_opacity(), 6))
        for mob in mobjects
    ]


def translucent_squares(n):
    return [Square().set_fill(opacity=0.2).set_stroke(opacity=0.5) for _ in range(n)]


def test_show_increasing_subsets_keeps_opacity():
    """Shown submobjects keep their own fill and stroke opacity (issue #3424)."""
    squares = VGroup(*translucent_squares(4))
    anim = ShowIncreasingSubsets(squares)
    anim.begin()
    anim.interpolate(0.5)
    assert opacities(squares) == [(0.2, 0.5)] * 2 + [(0, 0)] * 2
    anim.finish()
    assert opacities(squares) == [(0.2, 0.5)] * 4


def test_show_increasing_subsets_restores_opacity_attribute():
    """The fill_opacity attribute is restored too, as a number."""
    squares = VGroup(*translucent_squares(2))
    anim = ShowIncreasingSubsets(squares)
    anim.begin()
    anim.finish()
    assert [square.fill_opacity for square in squares] == [0.2, 0.2]


def test_show_increasing_subsets_with_overshooting_rate_function():
    """A rate function above 1 shows every submobject."""
    squares = VGroup(*translucent_squares(3))
    anim = ShowIncreasingSubsets(squares, rate_func=lambda t: 1.5)
    anim.begin()
    anim.interpolate(0.5)
    assert opacities(squares) == [(0.2, 0.5)] * 3


def test_remove_after_add_text_letter_by_letter_keeps_opacity():
    """Created before the first one ran, the second animation still knows the opacity."""
    text = Text("Hi", fill_opacity=0.5)
    add, remove = AddTextLetterByLetter(text), RemoveTextLetterByLetter(text)
    add.begin()
    add.finish()
    remove.begin()
    assert [round(letter.get_fill_opacity(), 6) for letter in text] == [0.5, 0.5]


def test_show_submobjects_one_by_one_keeps_opacity():
    """The shown submobject keeps its own fill and stroke opacity."""
    squares = translucent_squares(3)
    anim = ShowSubmobjectsOneByOne(squares)
    anim.begin()
    anim.interpolate(0.5)
    assert opacities(squares) == [(0, 0), (0.2, 0.5), (0, 0)]


def test_add_text_letter_by_letter_keeps_opacity():
    """Letters keep the opacity of the text."""
    text = Text("Hi", fill_opacity=0.5)
    anim = AddTextLetterByLetter(text)
    anim.begin()
    anim.finish()
    assert [round(letter.get_fill_opacity(), 6) for letter in text] == [0.5, 0.5]


def test_type_with_cursor_keeps_opacity():
    """Letters keep the opacity of the text when typed with a cursor."""
    text = Text("Hi", fill_opacity=0.5)
    letters = list(text)
    anim = TypeWithCursor(text, Rectangle(width=0.1, height=0.5))
    anim.begin()
    anim.finish()
    assert [round(letter.get_fill_opacity(), 6) for letter in letters] == [0.5, 0.5]
