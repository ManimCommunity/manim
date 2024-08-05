from manim import RED, MarkupText, Text, VMobject

__module_test__ = "text"


def test_Text2Color():
    txt = Text(
        "this is  a text  with spaces!",
        t2c={"spaces": RED},
        stroke_width=1,
        disable_ligatures=True,
    )
    assert len(txt.submobjects) == 29
    assert all(char.fill_color.to_hex() == "#FFFFFF" for char in txt[:4])  # "this"
    assert all(
        char.fill_color.to_hex() == RED.to_hex() for char in txt[-7:-1]
    )  # "spaces"
    assert txt[-1].fill_color.to_hex() == "#FFFFFF"  # "!"


def test_text_color_inheritance():
    """Test that Text and MarkupText correctly inherit colour from
    their parent class.
    """
    VMobject.set_default(color=RED)
    # set both to a singular font so that the tests agree.
    text = Text("test_color_inheritance", font="Sans")
    markup_text = MarkupText("test_color_inheritance", font="Sans")

    assert all(char.fill_color.to_hex() == RED.to_hex() for char in text)
    assert all(char.fill_color.to_hex() == RED.to_hex() for char in markup_text)

    # reset the default color so that future tests aren't affected by this change.
    VMobject.set_default()
