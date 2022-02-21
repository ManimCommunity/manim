import pytest

from manim import *
from manim.utils.testing.frames_comparison import frames_comparison

__module_test__ = "text"


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="MacOS and Windows render fonts differently, so they need separate comparison data.",
)
@frames_comparison
def test_Text2Color(scene):
    scene.add(Text("this is  a text  with spaces!", t2c={"spaces": RED}))


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="MacOS and Windows render fonts differently, so they need separate comparison data.",
)
@frames_comparison
def test_text_color_inheritance(scene):
    """Test that Text and MarkupText correctly inherit colour from
    their parent class."""
    VMobject.set_default(color=RED)
    # set both to a singular font so that the tests agree.
    text = Text("test_color_inheritance", font="Dejavu Sans")
    markup_text = MarkupText("test_color_inheritance", font="Dejavu Sans")
    vgr = VGroup(text, markup_text).arrange()

    # reset the default color so that future tests aren't affected by this change.
    VMobject.set_default()

    scene.add(vgr)
