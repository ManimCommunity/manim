import pytest

from manim.mobject.svg.text_mobject import MarkupText, Text
from manim import RED

def get_color(mobject):

    return mobject.get_color().get_hex()

def test_font_size():
    """Test that Text and MarkupText return the
    correct font_size value after being scaled."""
    text_string = Text("0").scale(0.3)
    markuptext_string = MarkupText("0").scale(0.3)

    assert round(text_string.font_size, 5) == 14.4
    assert round(markuptext_string.font_size, 5) == 14.4

def test_set_color_by_t2c():
    """Test Text.set_color_by_t2c()."""
    text = Text(
        "Cherry Blossoms in Spring", disable_ligatures=False, t2c={"Spring": RED}
    )
    assert all(get_color(c) == RED.lower() for c in text.submobjects[-6:])

    text.disable_ligatures = True
    assert all(get_color(c) == RED.lower() for c in text.submobjects[-6:])

    text2 = Text(
        "Cherry Blossoms in Spring", disable_ligatures=False, t2c={"Blossoms in": RED}
    )
    assert all(get_color(c) == RED.lower() for c in text2.submobjects[7:16])

    text2.disable_ligatures = True
    assert all(get_color(c) == RED.lower() for c in text2.submobjects[7:16])

    text3 = Text(
        "Cherry Blossoms in Spring Cherry Blossoms in Spring",
        disable_ligatures=False,
        t2c={"in Spring": RED},
    )
    assert all(get_color(c) == RED.lower() for c in text3.submobjects[-8:])
    assert all(get_color(c) == RED.lower() for c in text3.submobjects[14:22])

    text3.disable_ligatures = True
    assert all(get_color(c) == RED.lower() for c in text3.submobjects[-8:])
    assert all(get_color(c) == RED.lower() for c in text3.submobjects[14:22])
