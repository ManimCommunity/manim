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
        get_project_root() / "tests/test_graphical_units/img_svg_resources" / filename
    )


# Tests break down into two kinds: one where the SVG is simple enough to step through
# and ones where the SVG is realistically complex, and the output should be visually inspected.

# First are the simple tests.


class CubicPathTest(Scene):
    def construct(self):
        cubic_demo = SVGMobject(get_test_resource("cubic_demo.svg"))
        self.add(cubic_demo)
        self.wait()


class CubicAndLinetoTest(Scene):
    def construct(self):
        cubic_lineto = SVGMobject(get_test_resource("cubic_and_lineto.svg"))
        self.add(cubic_lineto)
        self.wait()


class RhomboidTest(Scene):
    pytest_skip = True

    def construct(self):
        # TODO: Fix behavior such that the polygon results in a closed shape, even without the closing z.
        # TODO: Discuss whether, upon loading an SVG, whether to obey the fill and stroke properties.
        rhomboid = SVGMobject(get_test_resource("rhomboid.svg")).shift(UP * 2)
        rhomboid_no_fill = (
            rhomboid.copy()
            .set_fill(opacity=0)
            .set_stroke(color=WHITE, width=1)
            .shift(DOWN * 4)
        )
        self.add(rhomboid, rhomboid_no_fill)
        self.wait()


class MultiPartPathTest(Scene):
    def construct(self):
        mpp = SVGMobject(get_test_resource("multi_part_path.svg"))
        self.add(mpp)
        self.wait()


class QuadraticPathTest(Scene):
    def construct(self):
        quad = SVGMobject(get_test_resource("qcurve_demo.svg"))
        self.add(quad)
        self.wait()


class SmoothCurvesTest(Scene):
    def construct(self):
        smooths = SVGMobject(get_test_resource("smooth_curves.svg")).set_fill(opacity=0)
        self.add(smooths)
        self.wait()


# TODO: test in a 3D Scene
class ThreeDSVGTest(ThreeDScene):
    pytest_skip = True


# Second are the visual tests - these are probably too complex to verify step-by-step, so
# these are really more of a spot-check


class WeightSVGTest(Scene):
    def construct(self):
        path = get_test_resource("weight.svg")
        svg_obj = SVGMobject(path)
        self.add(svg_obj)
        self.wait()


class BrachistochroneCurveTest(Scene):
    pytest_skip = True

    def construct(self):
        # TODO: There's a <rect> object with fill="none" that turns everything white.
        # TODO: the path has an implicit fill, too - even though the code says fill="none" and color="black"
        brach_curve = SVGMobject(get_test_resource("curve.svg")).set_fill(opacity=0)
        self.add(brach_curve)
        self.wait()


class DesmosGraph1Test(Scene):
    pytest_skip = True

    def construct(self):
        # TODO: white rect background, and stroke / fill colors aren't parsed.
        dgraph = SVGMobject(get_test_resource("desmos-graph_1.svg")).set_fill(opacity=0)
        self.add(dgraph)
        self.wait()


class Drawing4(Scene):
    pytest_skip = True

    def construct(self):
        # TODO: white hex background, and stroke / fill colors aren't parsed.
        draw4 = SVGMobject(get_test_resource("drawing4.svg"))
        self.add(draw4)
        self.wait()


class SingleUSStateTest(Scene):
    pytest_skip = True

    def construct(self):
        single_state = SVGMobject(get_test_resource("single_state.svg"))
        self.add(single_state)
        self.wait()


class ContiguousUSMapTest(Scene):
    pytest_skip = True

    def construct(self):
        states = SVGMobject(get_test_resource("states_map.svg"))
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
    GraphicalUnitTester(scene_to_test[1], MODULE_NAME, tmpdir).test(show_diff=True)
