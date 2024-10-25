from __future__ import annotations

import logging

import numpy as np

from manim import BackgroundRectangle, Circle, Sector, Square, SurroundingRectangle

logger = logging.getLogger(__name__)


def test_get_arc_center():
    np.testing.assert_array_equal(
        Sector(arc_center=[1, 2, 0]).get_arc_center(), [1, 2, 0]
    )


def test_SurroundingRectangle():
    circle = Circle()
    square = Square()
    sr = SurroundingRectangle(circle, square)
    sr.set_style(fill_opacity=0.42)
    assert sr.get_fill_opacity() == 0.42


def test_BackgroundRectangle(manim_caplog):
    circle = Circle()
    square = Square()
    bg = BackgroundRectangle(circle, square)
    bg.set_style(fill_opacity=0.42)
    assert bg.get_fill_opacity() == 0.42
    bg.set_style(fill_opacity=1, hello="world")
    assert (
        "Argument {'hello': 'world'} is ignored in BackgroundRectangle.set_style."
        in manim_caplog.text
    )


def test_Square_side_length_reflets_correct_width_and_height():
    sq = Square(side_length=1).scale(3)
    assert sq.side_length == 3
    assert sq.height == 3
    assert sq.width == 3


def test_changing_Square_side_length_updates_the_square_appropriately():
    sq = Square(side_length=1)
    sq.side_length = 3
    assert sq.height == 3
    assert sq.width == 3


def test_Square_side_length_consistent_after_scale_and_rotation():
    sq = Square(side_length=1).scale(3).rotate(np.pi / 4)
    assert np.isclose(sq.side_length, 3)
