import numpy as np
import pytest

from manim import Sector


def test_get_arc_center():
    assert np.all(Sector(arc_center=[1, 2, 0]).get_arc_center() == [1, 2, 0])
