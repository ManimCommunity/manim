import pytest

from manim import *

from ..utils.GraphicalUnitTester import GraphicalUnitTester
from ..utils.testing_utils import get_scenes_to_test


class TexTemplateFontTest(Scene):
    def construct(self):
        stix2 = TexTemplate()
        stix2.add_to_preamble(r"\usepackage{stix2}", prepend=True)

        def CustomMathTex(*tex_strings):
            return MathTex(*tex_strings, tex_template=stix2)

        tex = CustomMathTex(r"dx dy = |r| dr d\phi").scale(3)
        self.add(tex)


MODULE_NAME = "tex_and_text"


@pytest.mark.parametrize("scene_to_test", get_scenes_to_test(__name__), indirect=False)
def test_scene(scene_to_test, tmpdir, show_diff):
    GraphicalUnitTester(scene_to_test[1], MODULE_NAME, tmpdir).test(show_diff=show_diff)
