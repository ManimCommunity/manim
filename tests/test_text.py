from manim import Text


def test_long_string():
    """Check that `#469 <https://github.com/ManimCommunity/manim/issues/469>`_
    is resolved."""

    t = Text("a" * 150, font="Arial")
    assert len(t.text) == len(t.submobjects)
