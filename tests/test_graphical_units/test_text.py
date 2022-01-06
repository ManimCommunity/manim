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
