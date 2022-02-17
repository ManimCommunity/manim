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

# generated control data does not match control data produced locally on arch linux.
# possibly due to pango version mismatch?

# @frames_comparison
# def test_text_color_inheritance(scene):
#     """Test that Text and MarkupText correctly inherit colour from
#     their parent class."""
#
#     VMobject.set_default(color=RED)
#     text = Text("test_color_inheritance")
#     markup_text = MarkupText("test_color_inheritance")
#     vgr = VGroup(text, markup_text).arrange()
#
#     # reset the default color so that future tests aren't affected by this change.
#     VMobject.set_default()
#
#     scene.add(vgr)
