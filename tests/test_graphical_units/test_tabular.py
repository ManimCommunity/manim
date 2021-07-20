import pytest

from manim import *

from ..utils.GraphicalUnitTester import GraphicalUnitTester
from ..utils.testing_utils import get_scenes_to_test


class TabularTest(Scene):
    def construct(self):
        t = Tabular(
            [["1", "2"], ["3", "4"]],
            row_labels=[Tex("R1"), Tex("R2")],
            col_labels=[Tex("C1"), Tex("C2")],
            top_left_entry=MathTex("TOP"),
            element_to_mobject=Tex,
            include_outer_lines=True,
        )
        self.add(t)


class MathTabularTest(Scene):
    def construct(self):
        t = MathTabular([[1, 2], [3, 4]])
        self.add(t)


class MobjectTabularTest(Scene):
    def construct(self):
        a = Circle().scale(0.5)
        t = MobjectTabular([[a.copy(), a.copy()], [a.copy(), a.copy()]])
        self.add(t)


class IntegerTabularTest(Scene):
    def construct(self):
        t = IntegerTabular(
            np.arange(1, 21).reshape(5, 4),
        )
        self.add(t)


class DecimalTabularTest(Scene):
    def construct(self):
        t = DecimalTabular(
            np.linspace(0, 0.9, 20).reshape(5, 4),
        )
        self.add(t)


MODULE_NAME = "tabular"


@pytest.mark.parametrize("scene_to_test", get_scenes_to_test(__name__), indirect=False)
def test_scene(scene_to_test, tmpdir, show_diff):
    GraphicalUnitTester(scene_to_test[1], MODULE_NAME, tmpdir).test(show_diff=show_diff)
