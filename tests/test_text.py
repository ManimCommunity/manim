import pytest

from manim import BLUE, GREEN, RED, Text


def test_t2c():

    text1 = Text("fl ligature", t2c={"f": RED, "e": BLUE}, disable_ligatures=False)
    # If disable_ligatures is set to False, submobjects don't include blank spaces
    assert text1.submobjects[0].stroke_color == RED
    assert text1.submobjects[9].stroke_color == BLUE

    text2 = Text("fl ligature", t2c={"f": RED, "e": BLUE}, disable_ligatures=True)
    # If disable_ligatures is set to True, submobjects include blank spaces
    assert text2.submobjects[0].stroke_color == RED
    assert text2.submobjects[10].stroke_color == BLUE
