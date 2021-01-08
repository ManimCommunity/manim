import sys
from pathlib import Path

import pytest
from manim import *
from ..helpers.path_utils import get_project_root
from ..utils.testing_utils import get_scenes_to_test
from ..utils.GraphicalUnitTester import GraphicalUnitTester

from ..helpers.graphical_units import set_test_scene
# To update the reference object after a visual inspection, use this line:
# set_test_scene(NameOfTestScene, "img_and_svg")


def get_test_resource(filename):
    return str(
            get_project_root()
            / "tests/test_graphical_units/img_svg_resources"
            / filename
        )


class SVGMobjectTest(Scene):
    def construct(self):
        path = get_test_resource("weight.svg")
        svg_obj = SVGMobject(path)
        self.add(svg_obj)
        self.wait()


class RhomboidTest(Scene):
    def construct(self):
        # TODO: Fix behavior such that the polygon results in a closed shape, even without the closing z.
        # TODO: Discuss whether, upon loading an SVG, whether to obey the fill and stroke properties.
        rhomboid = SVGMobject(get_test_resource("rhomboid.svg")).shift(UP * 2)
        rhomboid_no_fill = rhomboid.copy().set_fill(opacity=0).set_stroke(color=WHITE, width=1).shift(DOWN * 4)
        self.add(rhomboid, rhomboid_no_fill)
        self.wait()


class SingleUSStateTest(Scene):
    def construct(self):
        states = SVGMobject(get_test_resource("single_state.svg"))
        self.add(states)
        self.wait()


class ImageMobjectTest(Scene):
    def construct(self):
        file_path = get_test_resource("tree_img_640x351.png")

        im1 = ImageMobject(file_path).shift(4 * LEFT + UP)
        im2 = ImageMobject(file_path, scale_to_resolution=1080).shift(
            4 * LEFT + 2 * DOWN
        )
        im3 = ImageMobject(file_path, scale_to_resolution=540).shift(4 * RIGHT)
        self.add(im1, im2, im3)
        self.wait(1)


MODULE_NAME = "img_and_svg"


@pytest.mark.parametrize("scene_to_test", get_scenes_to_test(__name__), indirect=False)
def test_scene(scene_to_test, tmpdir, show_diff):
    GraphicalUnitTester(scene_to_test[1], MODULE_NAME, tmpdir).test(show_diff=show_diff)
